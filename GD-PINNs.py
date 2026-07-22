import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.optim import LBFGS
from scipy.spatial import cKDTree
from scipy.fft import fftn, fftshift
from torch_geometric.nn import GATConv


# =============================================================================
# Stage 1: Spectral-Physical Manifold Parameterization
# =============================================================================

def nondimensionalize_coordinates(coords):
    """
    Normalizes coordinates by the characteristic length scale.
    Translates the domain to bound the physical manifold for PINN stability.
    Returns normalized coordinates and the characteristic length for PDE chain rule scaling.
    """
    coords_min = np.min(coords, axis=0)
    l_ref = np.max(np.max(coords, axis=0) - coords_min)
    coords_norm = (coords - coords_min) / l_ref
    return coords_norm, l_ref


def fast_idw_interpolation(coords_norm, fields, resolution=64):
    """
    Projects unstructured physical fields onto a unified Cartesian grid.
    Utilizes Inverse Distance Weighting (IDW) via k-d tree to preserve
    topological structures efficiently without memory bottleneck.
    """
    axis = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing='ij')
    target_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    tree = cKDTree(coords_norm)
    dists, indices = tree.query(target_points, k=3)

    weights = 1.0 / (dists**2 + 1e-8)
    weights /= np.sum(weights, axis=1, keepdims=True)

    interp_fields = np.sum(fields[indices] * weights[..., np.newaxis], axis=1)

    return interp_fields.reshape(resolution, resolution, resolution, fields.shape[1])


class SpectralPhysicalExtractor:
    """
    Constructs a 150-dimensional manifold embedding vector z_i.
    Combines Fast Fourier Transform (FFT), Power Spectral Density (PSD),
    and Proper Orthogonal Decomposition (POD) components.
    """

    def __init__(self, d_fft=80, d_psd=60, d_pod=10):
        self.d_fft = d_fft
        self.d_psd = d_psd
        self.d_pod = d_pod

    def extract_fft(self, tensor_3d):
        """
        Extracts dominant global transport organization via low-wavenumber FFT coefficients.
        Retains the central 5x5x5 low-frequency structural cube.
        """
        fft_features = []
        center = np.array(tensor_3d.shape[:3]) // 2

        for c in range(tensor_3d.shape[-1]):
            fft_3d = np.abs(fftn(tensor_3d[..., c]))
            fft_3d = fftshift(fft_3d)

            low_freq = fft_3d[center[0]-2:center[0]+3,
                              center[1]-2:center[1]+3,
                              center[2]-2:center[2]+3]
            sorted_modes = np.sort(low_freq.flatten())[::-1][:20]
            sorted_modes = np.pad(sorted_modes, (0, max(0, 20 - len(sorted_modes))))
            fft_features.extend(sorted_modes / (np.sum(sorted_modes) + 1e-8))

        return np.array(fft_features[:self.d_fft])

    def extract_psd(self, tensor_3d):
        """
        Computes multiscale spectral-energy distribution across independent spatial axes.
        """
        psd_features = []
        energy_field = np.sum(tensor_3d[..., :3]**2, axis=-1)

        for axis in range(3):
            axis_energy = np.sum(energy_field, axis=tuple([i for i in range(3) if i != axis]))
            psd_1d = np.abs(np.fft.fft(axis_energy))[:20]
            psd_1d = np.pad(psd_1d, (0, max(0, 20 - len(psd_1d))))[:20]
            psd_features.extend(psd_1d / (np.sum(psd_1d) + 1e-8))

        return np.array(psd_features[:self.d_psd])

    def extract_pod(self, tensor_3d):
        """
        Captures coherent physical deformation structures via Singular Value Decomposition.
        Slices the Cartesian tensor along the Z-axis to construct the spatial snapshot matrix.
        """
        res = tensor_3d.shape[0]
        snapshot = tensor_3d.transpose(2, 0, 1, 3).reshape(res, -1)

        try:
            _, S, _ = np.linalg.svd(snapshot, full_matrices=False)
            energy = S**2
            total_energy = np.sum(energy)

            cumsum = np.cumsum(energy) / total_energy
            k_95 = np.argmax(cumsum >= 0.95) + 1
            k_opt = min(k_95, self.d_pod)

            pod_features = energy[:k_opt] / total_energy
        except np.linalg.LinAlgError:
            pod_features = np.ones(self.d_pod) / self.d_pod

        return np.pad(pod_features, (0, max(0, self.d_pod - len(pod_features))))

    def compute_features(self, tensor_3d):
        """
        Concatenates FFT, PSD, and POD features into the final spectral-physical embedding.
        """
        fft_feat = self.extract_fft(tensor_3d)
        psd_feat = self.extract_psd(tensor_3d)
        pod_feat = self.extract_pod(tensor_3d)

        return np.concatenate([fft_feat, psd_feat, pod_feat])


# =============================================================================
# Stage 2: Adaptive Graph Initialization
# =============================================================================

class ManifoldGNN(nn.Module):
    """
    Implements anisotropic diffusion on the discrete solution manifold via Graph Attention.
    """
    def __init__(self, input_dim=150, hidden_dim=128, output_dim=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.leaky_relu(self.gat1(x, edge_index, edge_weight), negative_slope=0.2)
        x = F.leaky_relu(self.gat2(x, edge_index, edge_weight), negative_slope=0.2)
        return x


class GraphManifoldConstructor:
    """
    Builds the topological k-NN graph based on cosine similarity metrics.
    Executes adaptive Anchor Support Set expansion evaluated by structural covariance.
    """
    def __init__(self, device, k_neighbors=5, output_dim=64):
        self.device = device
        self.k_neighbors = k_neighbors
        self.output_dim = output_dim
        self.embeddings = None
        self.dense_adj = None

    def build_and_train_graph(self, features, epochs=200):
        """
        Constructs the sparse adjacency matrix with explicit self-loops.
        Trains the GNN to project descriptors into the latent manifold space.
        """
        X = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)

        norms = torch.norm(X, p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(X / (norms + 1e-8), (X / (norms + 1e-8)).t())

        actual_k = min(self.k_neighbors + 1, len(X))
        values, indices = torch.topk(sim_matrix, k=actual_k, dim=1)

        src, dst, w = [], [], []
        for i in range(len(X)):
            for j_idx, sim_val in zip(indices[i], values[i]):
                j = j_idx.item()
                if i != j:
                    src.append(i)
                    dst.append(j)
                    w.append(sim_val.item())

            # Explicitly enforce self-loops for GAT topology retention
            src.append(i)
            dst.append(i)
            w.append(1.0)

        edge_index = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        edge_weights = torch.tensor(w, dtype=torch.float32).to(self.device)

        self.dense_adj = torch.zeros((len(X), len(X)), device=self.device)
        self.dense_adj[edge_index[0], edge_index[1]] = edge_weights

        gnn = ManifoldGNN(input_dim=X.shape[1], output_dim=self.output_dim).to(self.device)
        optimizer = optim.Adam(gnn.parameters(), lr=0.01)

        for _ in range(epochs):
            optimizer.zero_grad()
            z = gnn(X, edge_index, edge_weights)
            z_norm = F.normalize(z, p=2, dim=1)

            loss = F.mse_loss(torch.matmul(z_norm, z_norm.t()), self.dense_adj)
            loss.backward()
            optimizer.step()

        gnn.eval()
        with torch.no_grad():
            self.embeddings = gnn(X, edge_index, edge_weights)

        return self.embeddings

    def expand_anchor_set(self, features, max_iter=100, tau_anchor=0.01):
        """
        Progressively samples the manifold topology until the Frobenius norm
        of the covariance increment satisfies the convergence threshold.
        """
        prev_cov = None

        for m in range(1, max_iter + 1):
            n_anchor = min(m * 10, len(features))

            if n_anchor >= len(features):
                return np.arange(len(features))

            anchor_indices = np.random.choice(len(features), n_anchor, replace=False)
            anchor_emb = self.embeddings[anchor_indices]

            cov = torch.cov(anchor_emb.T)

            if prev_cov is not None:
                diff_norm = torch.norm(cov - prev_cov, p='fro')
                prev_norm = torch.norm(prev_cov, p='fro') + 1e-8
                eta = diff_norm / prev_norm

                if eta.item() < tau_anchor:
                    return anchor_indices

            prev_cov = cov

        return np.arange(len(features))

    def select_next_task(self, trained_indices, remaining_indices):
        """
        Identifies the optimal subsequent transfer target based on
        multi-hop topological radiation across the manifold.
        """
        influence = torch.zeros(len(self.embeddings), device=self.device)
        influence[trained_indices] = 1.0

        for _ in range(3):
            influence = torch.matmul(self.dense_adj, influence)

        best_idx = max(remaining_indices, key=lambda idx: influence[idx].item())
        return best_idx

    def get_transfer_weights(self, target_idx, trained_indices, tau=0.1):
        """
        Determines the 1-NN task for strict operator inheritance and calculates
        softmax-normalized graph-proximity weights for adapter interpolation.
        """
        target_emb = self.embeddings[target_idx]
        candidate_embs = self.embeddings[trained_indices]

        sims = F.cosine_similarity(target_emb.unsqueeze(0), candidate_embs)

        k_actual = min(self.k_neighbors, len(trained_indices))
        _, top_k = torch.topk(sims, k_actual)
        neighbor_idx = [trained_indices[i] for i in top_k.cpu().numpy()]

        neighbor_embs = self.embeddings[neighbor_idx]
        logits = torch.matmul(neighbor_embs, target_emb) / tau
        weights = F.softmax(logits, dim=0).cpu().numpy()

        nearest_idx = neighbor_idx[0]

        return nearest_idx, neighbor_idx, weights


# =============================================================================
# Stage 3: Manifold-Conditioned Operator-Geometry Transfer
# =============================================================================

class GD_PINNs_Net(nn.Module):
    """
    Decoupled neural architecture separating invariant underlying PDE physics
    from geometry-specific variation functions via FiLM layer modulation.
    """
    def __init__(self, Re=229.1451, graph_emb_dim=64):
        super().__init__()
        self.Re = Re

        self.encoder = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh()
        )

        self.film_mlp = nn.Sequential(
            nn.Linear(graph_emb_dim, 100), nn.ReLU(),
            nn.Linear(100, 200)
        )

        self.adapter = nn.Sequential(
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 4)
        )

    def forward(self, x, h_graph):
        """
        Executes Feature-wise Linear Modulation conditioning.
        """
        latent = self.encoder(x)
        film_params = self.film_mlp(h_graph)
        gamma = film_params[:, :100]
        beta = film_params[:, 100:]

        modulated = gamma * latent + beta
        return self.adapter(modulated)

    def net_f_NS(self, x, h_graph, l_ref):
        """
        Evaluates Navier-Stokes physical residual constraints.
        Enforces proper partial derivative scaling strictly according to the chain rule
        dictated by spatial non-dimensionalization mappings.
        """
        x = x.clone().detach().requires_grad_(True)
        uvwp = self.forward(x, h_graph)
        u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]

        def grads(y):
            return autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        u_grads = grads(u)
        v_grads = grads(v)
        w_grads = grads(w)
        p_grads = grads(p)

        u_x = u_grads[:, 0:1] / l_ref
        u_y = u_grads[:, 1:2] / l_ref
        u_z = u_grads[:, 2:3] / l_ref

        v_x = v_grads[:, 0:1] / l_ref
        v_y = v_grads[:, 1:2] / l_ref
        v_z = v_grads[:, 2:3] / l_ref

        w_x = w_grads[:, 0:1] / l_ref
        w_y = w_grads[:, 1:2] / l_ref
        w_z = w_grads[:, 2:3] / l_ref

        p_x = p_grads[:, 0:1] / l_ref
        p_y = p_grads[:, 1:2] / l_ref
        p_z = p_grads[:, 2:3] / l_ref

        u_xx = grads(u_grads[:, 0:1])[:, 0:1] / (l_ref**2)
        u_yy = grads(u_grads[:, 1:2])[:, 1:2] / (l_ref**2)
        u_zz = grads(u_grads[:, 2:3])[:, 2:3] / (l_ref**2)

        v_xx = grads(v_grads[:, 0:1])[:, 0:1] / (l_ref**2)
        v_yy = grads(v_grads[:, 1:2])[:, 1:2] / (l_ref**2)
        v_zz = grads(v_grads[:, 2:3])[:, 2:3] / (l_ref**2)

        w_xx = grads(w_grads[:, 0:1])[:, 0:1] / (l_ref**2)
        w_yy = grads(w_grads[:, 1:2])[:, 1:2] / (l_ref**2)
        w_zz = grads(w_grads[:, 2:3])[:, 2:3] / (l_ref**2)

        inv_Re = 1.0 / self.Re

        f_e = u_x + v_y + w_z

        f_u = u * u_x + v * u_y + w * u_z + p_x - inv_Re * (u_xx + u_yy + u_zz)
        f_v = u * v_x + v * v_y + w * v_z + p_y - inv_Re * (v_xx + v_yy + v_zz)
        f_w = u * w_x + v * w_y + w * w_z + p_z - inv_Re * (w_xx + w_yy + w_zz)

        return uvwp, f_u, f_v, f_w, f_e


class OperatorGeometryTransfer:
    """
    Coordinates decoupled inheritance policies, spatial sub-sampling,
    model parameter optimization, and metric evaluation.
    """
    def __init__(self, device, lambda_b=50.0):
        self.device = device
        self.lambda_b = lambda_b

    def hot_start_target_model(self, target_model, source_state_1nn, neighbor_states, weights):
        """
        Executes structural fusion initialization protocols.
        Combines identical physics operator loading with linear manifold adapter blending.
        """
        target_state = target_model.state_dict()

        for k in target_state.keys():
            if 'encoder' in k and k in source_state_1nn:
                target_state[k].copy_(source_state_1nn[k])
            elif 'adapter' in k:
                fused = torch.zeros_like(target_state[k])
                for w, state in zip(weights, neighbor_states):
                    if k in state:
                        fused += w * state[k]
                target_state[k].copy_(fused)

        target_model.load_state_dict(target_state, strict=False)
        return target_model

    def topology_preserving_subsample(self, coords, fields, retain_ratio=0.3):
        """
        Conducts voxel-based spatial filtering to extract a structurally representative
        dataset subset for sparse data transfer task supervision.
        """
        num_points = len(coords)
        target_points = int(num_points * retain_ratio)

        if target_points <= 0 or target_points >= num_points:
            return coords, fields

        extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        volume = np.prod(np.where(extent > 0, extent, 1e-8))
        voxel_size = (volume / target_points) ** (1.0 / 3.0)

        voxel_idx = np.floor((coords - np.min(coords, axis=0)) / voxel_size).astype(int)
        hashes = voxel_idx[:, 0] + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2] * 1000000

        df = pd.DataFrame({'hash': hashes, 'idx': np.arange(num_points)})
        sampled_idx = df.groupby('hash').first()['idx'].values

        if len(sampled_idx) < target_points:
            remaining = target_points - len(sampled_idx)
            available = np.setdiff1d(np.arange(num_points), sampled_idx)
            topup = np.random.choice(available, remaining, replace=False)
            sampled_idx = np.concatenate([sampled_idx, topup])

        return coords[sampled_idx], fields[sampled_idx]

    def evaluate_model(self, model, X_test, Y_test, h_graph):
        """
        Computes the academic relative L2 error norm mapping prediction discrepancies
        against ground truth field magnitudes.
        """
        model.eval()
        with torch.no_grad():
            pred = model(X_test, h_graph)
            error = torch.norm(pred - Y_test, p=2, dim=0)
            norm = torch.norm(Y_test, p=2, dim=0) + 1e-8
            rel_l2 = error / norm
        return rel_l2.mean().item()

    def train_pinn(self, model, coords, fields, l_ref, h_graph, is_anchor=False, coarse_ratio=0.3):
        """
        Executes sequential Adam and L-BFGS-B loss minimization algorithms
        combining boundary term supervision and unsupervised PDE residual constraint.
        """
        if not is_anchor:
            coords_sampled, fields_sampled = self.topology_preserving_subsample(
                coords, fields, coarse_ratio
            )
        else:
            coords_sampled, fields_sampled = coords, fields

        X_phys = torch.tensor(coords, dtype=torch.float32, requires_grad=True).to(self.device)
        X_bound = torch.tensor(coords_sampled, dtype=torch.float32).to(self.device)
        Y_bound = torch.tensor(fields_sampled, dtype=torch.float32).to(self.device)

        # Test tensors mapped for Relative L2 evaluation
        X_test = torch.tensor(coords, dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(fields, dtype=torch.float32).to(self.device)

        h_graph_phys = h_graph.unsqueeze(0).repeat(X_phys.shape[0], 1)
        h_graph_bound = h_graph.unsqueeze(0).repeat(X_bound.shape[0], 1)
        h_graph_test = h_graph.unsqueeze(0).repeat(X_test.shape[0], 1)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        n_epochs = 10000 if is_anchor else 4000
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            pred_bound = model(X_bound, h_graph_bound)
            loss_data = torch.mean((pred_bound - Y_bound) ** 2)

            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_phys, l_ref)
            loss_pde = torch.mean(f_u**2 + f_v**2 + f_w**2 + f_e**2)

            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            optimizer.step()

        lbfgs = LBFGS(model.parameters(), max_iter=2000,
                      tolerance_grad=1e-5, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            pred_bound = model(X_bound, h_graph_bound)
            loss_data = torch.mean((pred_bound - Y_bound) ** 2)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_phys, l_ref)
            loss_pde = torch.mean(f_u**2 + f_v**2 + f_w**2 + f_e**2)
            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            return loss

        lbfgs.step(closure)

        rel_l2 = self.evaluate_model(model, X_test, Y_test, h_graph_test)
        print(f"    Optimization Complete | Relative L2 Error: {rel_l2:.4e}")

        return model


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GD-PINNs: Manifold-Guided Transfer Learning')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--lambda_b', type=float, default=50.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    extractor = SpectralPhysicalExtractor()
    graph_constructor = GraphManifoldConstructor(device, k_neighbors=args.k_neighbors)
    transfer_engine = OperatorGeometryTransfer(device, lambda_b=args.lambda_b)

    # ========================================================================
    # Stage 1: Spectral-Physical Embedding
    # ========================================================================
    features = []
    task_tensors = {}
    task_l_ref = {}
    task_ids = {}

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, 'case_*.csv')))

    for idx, csv_path in enumerate(csv_files):
        task_id = int(os.path.basename(csv_path).replace('case_', '').replace('.csv', ''))

        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values
        fields = df[['u', 'v', 'w', 'p']].values

        coords_norm, l_ref = nondimensionalize_coordinates(coords)
        tensor_3d = fast_idw_interpolation(coords_norm, fields, args.resolution)

        z_i = extractor.compute_features(tensor_3d)
        features.append(z_i)

        task_tensors[idx] = (coords_norm, fields)
        task_l_ref[idx] = l_ref
        task_ids[idx] = task_id

    features = np.array(features)

    # ========================================================================
    # Stage 2: Adaptive Graph Initialization
    # ========================================================================
    embeddings = graph_constructor.build_and_train_graph(features)
    anchor_indices = graph_constructor.expand_anchor_set(features)

    # ========================================================================
    # Stage 3: Baseline Training for Anchor Support Set
    # ========================================================================
    for anchor_idx in anchor_indices:
        anchor_model = GD_PINNs_Net().to(device)
        h_anchor = embeddings[anchor_idx].detach()

        coords, fields = task_tensors[anchor_idx]
        l_ref = task_l_ref[anchor_idx]

        anchor_model = transfer_engine.train_pinn(
            anchor_model, coords, fields, l_ref, h_anchor, is_anchor=True
        )
        torch.save(anchor_model.state_dict(),
                   os.path.join(args.output_dir, f'task_{task_ids[anchor_idx]}_anchor.pth'))

    # ========================================================================
    # Stage 4: Operator-Geometry Decoupled Progressive Transfer
    # ========================================================================
    trained_indices = list(anchor_indices)
    remaining_indices = [i for i in range(len(features)) if i not in trained_indices]

    while remaining_indices:
        target_idx = graph_constructor.select_next_task(trained_indices, remaining_indices)
        task_id = task_ids[target_idx]

        nearest_idx, neighbor_idx, weights = graph_constructor.get_transfer_weights(
            target_idx, trained_indices
        )

        neighbor_states = []
        for ni in neighbor_idx:
            nid = task_ids[ni]
            migrated_path = os.path.join(args.output_dir, f'task_{nid}_migrated.pth')
            anchor_path = os.path.join(args.output_dir, f'task_{nid}_anchor.pth')
            load_path = migrated_path if os.path.exists(migrated_path) else anchor_path
            neighbor_states.append(torch.load(load_path, map_location=device))

        state_1nn = neighbor_states[neighbor_idx.index(nearest_idx)]

        target_model = GD_PINNs_Net().to(device)
        target_model = transfer_engine.hot_start_target_model(
            target_model, state_1nn, neighbor_states, weights
        )

        h_target = embeddings[target_idx].detach()
        coords, fields = task_tensors[target_idx]
        l_ref = task_l_ref[target_idx]

        target_model = transfer_engine.train_pinn(
            target_model, coords, fields, l_ref, h_target,
            is_anchor=False, coarse_ratio=0.3
        )

        torch.save(target_model.state_dict(),
                   os.path.join(args.output_dir, f'task_{task_id}_migrated.pth'))

        trained_indices.append(target_idx)
        remaining_indices.remove(target_idx)


if __name__ == '__main__':
    main()
