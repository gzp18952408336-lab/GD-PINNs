"""
GD-PINNs: Graph-Diffusion Physics-Informed Neural Networks
Reference: "A spectral-physical manifold for scalable knowledge transfer in partial differential equation families"
Description: End-to-end pipeline including Cartesian resampling, 3D spectral feature extraction,
             manifold-conditioned architecture (FiLM), and graph-guided decoupled transfer.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.optim import LBFGS
from scipy.interpolate import griddata
from scipy.fft import fftn, fftshift

# PyG Imports
from torch_geometric.nn import GATConv


# =============================================================================
# 1. Data Preprocessing & Cartesian Resampling
# Ref: Supplementary Note 2.2
# =============================================================================
def nondimensionalize_coordinates(coords):
    """Maps spatial coordinates to [-1, 1]^3 for active tanh gradients."""
    coords_min = np.min(coords, axis=0)
    l_ref = np.max(np.max(coords, axis=0) - coords_min)
    coords_norm = 2.0 * (coords - coords_min) / l_ref - 1.0
    return np.clip(coords_norm, -1.0, 1.0)


def cartesian_resampling(coords_norm, physical_fields, resolution=32):
    """
    Interpolates unstructured data onto a unified 3D Cartesian grid.
    This eliminates mesh-dependent artifacts for subsequent 3D FFT.
    """
    grid_1d = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    target_points = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T

    num_channels = physical_fields.shape[1]
    structured_fields = np.zeros((target_points.shape[0], num_channels))

    for c in range(num_channels):
        # Linear interpolation preserves dominant low-frequency F-principle structures
        interpolated_c = griddata(coords_norm, physical_fields[:, c], target_points, method='linear')
        if np.isnan(interpolated_c).any():
            nearest_c = griddata(coords_norm, physical_fields[:, c], target_points, method='nearest')
            interpolated_c = np.where(np.isnan(interpolated_c), nearest_c, interpolated_c)
        structured_fields[:, c] = interpolated_c

    return structured_fields.reshape((resolution, resolution, resolution, num_channels))


# =============================================================================
# 2. 3D Spectral-Physical Feature Extraction
# Ref: Main Text Section 4.1 & Supplementary Note 2.4
# =============================================================================
class SpectralPhysicalExtractor:
    def __init__(self, d_pod=10, d_psd=60, d_fft=80):
        self.d_pod = d_pod
        self.d_psd = d_psd
        self.d_fft = d_fft

    def compute_features(self, tensor_3d):
        """Extracts 150-dimensional physical manifold coordinates from structured 3D tensors."""
        fft_features = []
        for c in range(4):  # u, v, w, p
            fft_3d = np.abs(fftn(tensor_3d[..., c]))
            fft_3d = fftshift(fft_3d)
            center = np.array(fft_3d.shape) // 2
            sub_vol = fft_3d[center[0] - 2:center[0] + 3, center[1] - 2:center[1] + 3, center[2] - 2:center[2] + 3]
            top_modes = np.sort(sub_vol.flatten())[::-1][:20]
            fft_features.extend(top_modes / (np.sum(top_modes) + 1e-8))

        psd_features = []
        for axis in range(3):
            axis_energy = np.sum(tensor_3d[..., 0] ** 2 + tensor_3d[..., 1] ** 2 + tensor_3d[..., 2] ** 2,
                                 axis=tuple([i for i in range(3) if i != axis]))
            psd_1d = np.abs(np.fft.fft(axis_energy))[:20]
            psd_1d = np.pad(psd_1d, (0, max(0, 20 - len(psd_1d))))[:20]
            psd_features.extend(psd_1d / (np.sum(psd_1d) + 1e-8))

        snapshot_matrix = tensor_3d.reshape(-1, 4)
        try:
            _, S, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
            energy = S ** 2
            pod_features = (energy / np.sum(energy))[:self.d_pod]
        except np.linalg.LinAlgError:
            pod_features = np.ones(self.d_pod) / self.d_pod
        pod_features = np.pad(pod_features, (0, max(0, self.d_pod - len(pod_features))))

        z_i = np.concatenate([fft_features, psd_features, pod_features])
        return z_i


# =============================================================================
# 3. Manifold-Conditioned Operator-Geometry Architecture
# Ref: Main Text Section 4.3 & Supplementary Note 2.3
# =============================================================================
class GD_PINNs_Net(nn.Module):
    def __init__(self, Re=229.1451, graph_emb_dim=64):
        super().__init__()
        self.Re = Re

        # Locally Inherited Operator Encoder (Invariant Block) - Width 100
        self.encoder = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh()
        )

        # FiLM Generator (Eq. 21) - Maps graph embedding to scaling & shifting coefficients
        self.film_mlp = nn.Sequential(
            nn.Linear(graph_emb_dim, 100), nn.ReLU(),
            nn.Linear(100, 200)  # Output 100 for gamma, 100 for beta
        )

        # Geometric Manifold Adapter (Variant Block) - Width 100
        self.adapter = nn.Sequential(
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 4)  # u, v, w, p
        )

    def forward(self, x, h_graph):
        latent_features = self.encoder(x)
        film_params = self.film_mlp(h_graph)
        gamma = film_params[:, :100]
        beta = film_params[:, 100:]

        # Feature-wise Linear Modulation
        modulated_features = gamma * latent_features + beta
        return self.adapter(modulated_features)

    def net_f_NS(self, x, h_graph):
        x = x.clone().detach().requires_grad_(True)
        uvwp = self.forward(x, h_graph)
        u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]

        def grads(y):
            return autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

        u_grads, v_grads, w_grads, p_grads = grads(u), grads(v), grads(w), grads(p)
        u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
        v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
        w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
        p_x, p_y, p_z = p_grads[:, 0:1], p_grads[:, 1:2], p_grads[:, 2:3]

        u_xx, u_yy, u_zz = grads(u_x)[:, 0:1], grads(u_y)[:, 1:2], grads(u_z)[:, 2:3]
        v_xx, v_yy, v_zz = grads(v_x)[:, 0:1], grads(v_y)[:, 1:2], grads(v_z)[:, 2:3]
        w_xx, w_yy, w_zz = grads(w_x)[:, 0:1], grads(w_y)[:, 1:2], grads(w_z)[:, 2:3]

        inv_Re = 1.0 / self.Re
        f_e = u_x + v_y + w_z
        f_u = u * u_x + v * u_y + w * u_z + p_x - inv_Re * (u_xx + u_yy + u_zz)
        f_v = u * v_x + v * v_y + w * v_z + p_y - inv_Re * (v_xx + v_yy + v_zz)
        f_w = u * w_x + v * w_y + w * w_z + p_z - inv_Re * (w_xx + w_yy + w_zz)

        return uvwp, f_u, f_v, f_w, f_e


# =============================================================================
# 4. Graph Architecture & Knowledge Network
# Ref: Supplementary Note 2.5
# =============================================================================
class ManifoldGNN(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=128, output_dim=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.leaky_relu(self.gat1(x, edge_index, edge_weight), negative_slope=0.2)
        x = F.leaky_relu(self.gat2(x, edge_index, edge_weight), negative_slope=0.2)
        return x


class AdvancedKnowledgeNetwork:
    def __init__(self, device, k_neighbors=5, output_dim=64):
        self.device = device
        self.k_neighbors = k_neighbors
        self.gnn = ManifoldGNN(output_dim=output_dim).to(device)
        self.task_embeddings = None

    def build_and_train_graph(self, features, epochs=200):
        print("[Graph] Constructing K-NN Manifold Graph...")
        X = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)

        norms = torch.norm(X, p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(X / (norms + 1e-8), (X / (norms + 1e-8)).t())
        num_nodes = len(X)
        actual_k = min(self.k_neighbors + 1, num_nodes)
        values, indices = torch.topk(sim_matrix, k=actual_k, dim=1)

        edge_src, edge_dst, edge_w = [], [], []
        for i in range(num_nodes):
            for j_idx, sim_val in zip(indices[i], values[i]):
                j = j_idx.item()
                if i != j:
                    edge_src.append(i), edge_dst.append(j), edge_w.append(sim_val.item())

        self.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(self.device)
        self.edge_weights = torch.tensor(edge_w, dtype=torch.float32).to(self.device)

        self.dense_adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        self.dense_adj[self.edge_index[0], self.edge_index[1]] = self.edge_weights

        print("[GNN] Training Manifold Representation...")
        optimizer = optim.Adam(self.gnn.parameters(), lr=0.01)
        for _ in range(epochs):
            optimizer.zero_grad()
            z = self.gnn(X, self.edge_index, self.edge_weights)
            z_norm = F.normalize(z, p=2, dim=1)
            loss = F.mse_loss(torch.matmul(z_norm, z_norm.t()), self.dense_adj)
            loss.backward()
            optimizer.step()

        self.gnn.eval()
        with torch.no_grad():
            self.task_embeddings = self.gnn(X, self.edge_index, self.edge_weights)
        return self.task_embeddings

    def find_anchor_task(self, alpha=0.65):
        """Eq. 9: Composite Centrality for Anchor Selection."""
        num_nodes = len(self.task_embeddings)
        degree = torch.zeros(num_nodes, device=self.device)
        src = self.edge_index[0]
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg_centrality = degree / (degree.max() + 1e-8)

        eig_centrality = torch.ones(num_nodes, device=self.device)
        for _ in range(20):
            eig_centrality = torch.matmul(self.dense_adj, eig_centrality)
            eig_centrality = eig_centrality / (torch.norm(eig_centrality) + 1e-8)

        c_center = alpha * eig_centrality + (1 - alpha) * deg_centrality
        return torch.argmax(c_center).item()

    def select_next_task(self, trained_indices, remaining_indices):
        influence = torch.zeros(len(self.task_embeddings), device=self.device)
        influence[trained_indices] = 1.0
        for _ in range(3):
            influence = torch.matmul(self.dense_adj, influence)

        best_idx = max(remaining_indices, key=lambda idx: influence[idx].item())
        return best_idx, influence[best_idx].item()

    def get_transfer_components(self, target_idx, trained_indices, tau=0.1):
        """Eq. 10: Graph-weighted parameter interpolation."""
        target_emb = self.task_embeddings[target_idx]
        candidates_emb = self.task_embeddings[trained_indices]
        sims = F.cosine_similarity(target_emb.unsqueeze(0), candidates_emb)

        k_actual = min(self.k_neighbors, len(trained_indices))
        _, top_k_local = torch.topk(sims, k_actual)
        neighbor_indices = [trained_indices[i] for i in top_k_local.cpu().numpy()]

        neighbor_embs = self.task_embeddings[neighbor_indices]
        weights = F.softmax(torch.matmul(neighbor_embs, target_emb) / tau, dim=0).cpu().numpy()
        return neighbor_indices, weights


# =============================================================================
# 5. Optimization & Transfer Engine
# Ref: Main Text Section 4.4 & Supplementary Note 2.7
# =============================================================================
class TransferTuner:
    def __init__(self, device, lambda_b=50.0):
        self.device = device
        self.lambda_b = lambda_b  # Strictly 50.0 for 3D topologies

    def execute_transfer(self, target_model, neighbor_indices, weights, model_dir, task_map):
        target_state = target_model.state_dict()
        fused_adapter_state = {k: torch.zeros_like(v).to(self.device) for k, v in target_state.items() if
                               'adapter' in k}

        nearest_idx = neighbor_indices[0]  # 1-NN Hard Transfer Base
        encoder_loaded = False

        for idx, graph_idx in enumerate(neighbor_indices):
            model_path = os.path.join(model_dir, f"task_{task_map[graph_idx]}_migrated.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(model_dir, f"task_{task_map[graph_idx]}_NS.pth")

            if os.path.exists(model_path):
                src_state = torch.load(model_path, map_location=self.device)

                # Invariant Block: 1-NN Inheritance
                if graph_idx == nearest_idx and not encoder_loaded:
                    for k in target_state.keys():
                        if 'encoder' in k: target_state[k].copy_(src_state[k])
                    encoder_loaded = True

                # Variant Block: Graph-weighted Interpolation
                for k in fused_adapter_state.keys():
                    fused_adapter_state[k] += weights[idx] * src_state[k]

        target_model.load_state_dict(fused_adapter_state, strict=False)
        return target_model

    def train_pinn(self, model, data, h_graph, is_anchor=False):
        """Hybrid Optimization using 30% Coarse-Grid Supervision."""
        coarse_ratio = 1.0 if is_anchor else 0.3  # 30% for topology-preserving transfer
        n_sample = max(1, int(len(data) * coarse_ratio))
        indices = np.random.choice(len(data), n_sample, replace=False)
        sampled_data = data[indices]

        # Use full available point cloud for physics collocation
        X_phys = torch.tensor(data[:, 0:3], dtype=torch.float32, requires_grad=True).to(self.device)
        X_bound = torch.tensor(sampled_data[:, 0:3], dtype=torch.float32).to(self.device)
        Y_bound = torch.tensor(sampled_data[:, 3:7], dtype=torch.float32).to(self.device)

        h_graph_expanded = h_graph.unsqueeze(0).repeat(X_phys.shape[0], 1)
        h_graph_bound = h_graph.unsqueeze(0).repeat(X_bound.shape[0], 1)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        epochs_adam = 10000 if is_anchor else 4000
        for epoch in range(epochs_adam):
            optimizer.zero_grad()
            pred_b = model(X_bound, h_graph_bound)
            loss_data = torch.mean((pred_b - Y_bound) ** 2)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_expanded)
            loss_pde = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2 + f_e ** 2)
            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            optimizer.step()

        print("[Tuner] Entering L-BFGS-B Refinement phase...")
        lbfgs = LBFGS(model.parameters(), max_iter=2000, tolerance_grad=1e-5, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            loss_data = torch.mean((model(X_bound, h_graph_bound) - Y_bound) ** 2)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_expanded)
            loss_pde = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2 + f_e ** 2)
            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            return loss

        lbfgs.step(closure)
        return model


# =============================================================================
# 6. Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output_models")
    parser.add_argument("--alpha", type=float, default=0.65, help="Centrality weight (0.65 SWRO, 0.50 Benchmarks)")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    extractor = SpectralPhysicalExtractor()
    knowledge_net = AdvancedKnowledgeNetwork(DEVICE, k_neighbors=5)
    tuner = TransferTuner(DEVICE, lambda_b=50.0)

    # Step 1: Preprocessing & Feature Extraction
    task_features, task_data_map, task_id_map = [], {}, {}
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "case_*.csv")))

    for idx, csv_path in enumerate(csv_files):
        print(f"[Process] Loading and mapping {csv_path}...")
        case_id = int(os.path.basename(csv_path).replace("case_", "").replace(".csv", ""))
        df = pd.read_csv(csv_path)

        coords = df[['x', 'y', 'z']].values
        phys_fields = df[['u', 'v', 'w', 'p']].values

        # Norm & Cartesian Resampling -> 3D Tensor -> Spectral Features
        coords_norm = nondimensionalize_coordinates(coords)
        tensor_3d = cartesian_resampling(coords_norm, phys_fields, resolution=32)
        feats = extractor.compute_features(tensor_3d)

        # Combine norm coords and fields for PINN training
        norm_data = np.hstack([coords_norm, phys_fields])

        task_features.append(feats)
        task_data_map[idx] = norm_data
        task_id_map[idx] = case_id

    # Step 2: Build Manifold Graph
    embeddings = knowledge_net.build_and_train_graph(task_features)
    anchor_idx = knowledge_net.find_anchor_task(alpha=args.alpha)
    print(f"\n[System] Identified Anchor Task ID: {task_id_map[anchor_idx]}")

    # Step 3: Train Anchor Task
    anchor_model = GD_PINNs_Net().to(DEVICE)
    h_anchor = embeddings[anchor_idx].detach()
    anchor_model = tuner.train_pinn(anchor_model, task_data_map[anchor_idx], h_anchor, is_anchor=True)
    torch.save(anchor_model.state_dict(), os.path.join(args.output_dir, f"task_{task_id_map[anchor_idx]}_NS.pth"))

    # Step 4: Progressive Transfer Execution
    trained_indices = [anchor_idx]
    remaining_indices = [i for i in range(len(task_features)) if i != anchor_idx]

    while remaining_indices:
        target_idx, _ = knowledge_net.select_next_task(trained_indices, remaining_indices)
        real_id = task_id_map[target_idx]
        h_target = embeddings[target_idx].detach()

        neighbor_indices, weights = knowledge_net.get_transfer_components(target_idx, trained_indices)
        print(f"\n[Transfer] Target ID {real_id} | Neighbors: {[task_id_map[i] for i in neighbor_indices]}")

        target_model = GD_PINNs_Net().to(DEVICE)
        target_model = tuner.execute_transfer(target_model, neighbor_indices, weights, args.output_dir, task_id_map)
        target_model = tuner.train_pinn(target_model, task_data_map[target_idx], h_target, is_anchor=False)
        """
        GD-PINNs: Graph-Diffusion Physics-Informed Neural Networks
        End-to-end pipeline with rigorous topological preservation and natural-neighbor surrogates.
        """

        import argparse
        import os
        import glob
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
        # 1. Rigorous Data Preprocessing & Topological Subsampling
        # =============================================================================
        def nondimensionalize_coordinates(coords):
            """Affine mapping to [-1, 1]^3 for optimal tanh active region gradients."""
            coords_min = np.min(coords, axis=0)
            l_ref = np.max(np.max(coords, axis=0) - coords_min)
            coords_norm = 2.0 * (coords - coords_min) / l_ref - 1.0
            return np.clip(coords_norm, -1.0, 1.0)


        def natural_neighbor_surrogate_resampling(coords_norm, solution_fields, resolution=64, k_neighbors=8):
            """
            Reconstructs continuous manifolds using KD-Tree IDW with adaptive Gaussian kernel.
            Replaces griddata(linear) to eliminate C1 discontinuities.
            """
            reference_axis = np.linspace(-1.0, 1.0, resolution)
            grid_x, grid_y, grid_z = np.meshgrid(reference_axis, reference_axis, reference_axis, indexing="ij")
            target_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

            tree = cKDTree(coords_norm)
            distances, indices = tree.query(target_points, k=k_neighbors)

            # Adaptive bandwidth based on Cartesian grid voxel size
            sigma = 2.0 / resolution
            weights = np.exp(-(distances ** 2) / (sigma ** 2)) + 1e-8
            weights /= np.sum(weights, axis=1, keepdims=True)

            num_channels = solution_fields.shape[1]
            structured_fields = np.zeros((target_points.shape[0], num_channels), dtype=np.float64)

            for c in range(num_channels):
                structured_fields[:, c] = np.sum(weights * solution_fields[indices, c], axis=1)

            return structured_fields.reshape(resolution, resolution, resolution, num_channels)


        def topology_preserving_subsampling(coords, solution_fields, retain_ratio=0.3):
            """
            Voxel Grid Filter for uniform spatial downsampling.
            Strictly preserves the spatial lattice structure required for topological anchoring.
            """
            num_points = len(coords)
            target_points = int(num_points * retain_ratio)
            if target_points <= 0 or target_points >= num_points:
                return coords, solution_fields

            domain_extent = np.max(coords, axis=0) - np.min(coords, axis=0)
            domain_volume = np.prod(np.where(domain_extent > 0, domain_extent, 1e-8))
            voxel_size = (domain_volume / target_points) ** (1 / 3)

            voxel_indices = np.floor((coords - np.min(coords, axis=0)) / voxel_size).astype(int)
            hashes = voxel_indices[:, 0] + voxel_indices[:, 1] * 1000 + voxel_indices[:, 2] * 1000000
            df = pd.DataFrame({'hash': hashes, 'index': np.arange(num_points)})

            sampled_indices = df.groupby('hash').first()['index'].values

            if len(sampled_indices) < target_points:
                remaining_needed = target_points - len(sampled_indices)
                available_indices = np.setdiff1d(np.arange(num_points), sampled_indices)
                top_up = np.random.choice(available_indices, remaining_needed, replace=False)
                sampled_indices = np.concatenate([sampled_indices, top_up])

            return coords[sampled_indices], solution_fields[sampled_indices]


        # =============================================================================
        # 2. 3D Spectral-Physical Feature Extraction
        # =============================================================================
        class SpectralPhysicalExtractor:
            def __init__(self, d_pod=10, d_psd=60, d_fft=80):
                self.d_pod = d_pod
                self.d_psd = d_psd
                self.d_fft = d_fft

            def compute_features(self, tensor_3d):
                fft_features = []
                center = np.array(tensor_3d.shape[:3]) // 2

                # Spherical Low-Pass Mask (R=3) to isolate pure low wavenumbers
                R = 3.0
                z_grid, y_grid, x_grid = np.ogrid[:tensor_3d.shape[0], :tensor_3d.shape[1], :tensor_3d.shape[2]]
                spherical_mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 + (
                            z_grid - center[2]) ** 2 <= R ** 2

                for c in range(4):
                    fft_3d = np.abs(fftn(tensor_3d[..., c]))
                    fft_3d = fftshift(fft_3d)
                    low_freq_modes = fft_3d[spherical_mask]
                    top_modes = np.sort(low_freq_modes)[::-1][:20]
                    top_modes = np.pad(top_modes, (0, max(0, 20 - len(top_modes))))
                    fft_features.extend(top_modes / (np.sum(top_modes) + 1e-8))

                psd_features = []
                for axis in range(3):
                    axis_energy = np.sum(tensor_3d[..., 0] ** 2 + tensor_3d[..., 1] ** 2 + tensor_3d[..., 2] ** 2,
                                         axis=tuple([i for i in range(3) if i != axis]))
                    psd_1d = np.abs(np.fft.fft(axis_energy))[:20]
                    psd_1d = np.pad(psd_1d, (0, max(0, 20 - len(psd_1d))))[:20]
                    psd_features.extend(psd_1d / (np.sum(psd_1d) + 1e-8))

                # Adaptive POD Truncation (>95% Energy)
                snapshot_matrix = tensor_3d.reshape(-1, 4)
                try:
                    _, S, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
                    energy = S ** 2
                    total_energy = np.sum(energy)
                    cumulative_energy = np.cumsum(energy) / total_energy
                    k_95 = np.argmax(cumulative_energy >= 0.95) + 1
                    k_optimal = min(k_95, self.d_pod)
                    pod_features = (energy[:k_optimal] / total_energy)
                except np.linalg.LinAlgError:
                    pod_features = np.ones(self.d_pod) / self.d_pod

                pod_features = np.pad(pod_features, (0, max(0, self.d_pod - len(pod_features))))

                return np.concatenate([fft_features, psd_features, pod_features])


        # =============================================================================
        # 3. Manifold-Conditioned Operator-Geometry Architecture (Width: 100)
        # =============================================================================
        class GD_PINNs_Net(nn.Module):
            def __init__(self, Re=229.1451, graph_emb_dim=64):
                super().__init__()
                self.Re = Re
                self.encoder = nn.Sequential(
                    nn.Linear(3, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh(),
                    nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh()
                )
                self.film_mlp = nn.Sequential(nn.Linear(graph_emb_dim, 100), nn.ReLU(), nn.Linear(100, 200))
                self.adapter = nn.Sequential(
                    nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 4)
                )

            def forward(self, x, h_graph):
                latent = self.encoder(x)
                film_params = self.film_mlp(h_graph)
                return self.adapter(film_params[:, :100] * latent + film_params[:, 100:])

            def net_f_NS(self, x, h_graph):
                x = x.clone().detach().requires_grad_(True)
                uvwp = self.forward(x, h_graph)
                u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]

                def grads(y): return autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

                u_grads, v_grads, w_grads, p_grads = grads(u), grads(v), grads(w), grads(p)
                u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
                v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
                w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
                p_x, p_y, p_z = p_grads[:, 0:1], p_grads[:, 1:2], p_grads[:, 2:3]

                u_xx, u_yy, u_zz = grads(u_x)[:, 0:1], grads(u_y)[:, 1:2], grads(u_z)[:, 2:3]
                v_xx, v_yy, v_zz = grads(v_x)[:, 0:1], grads(v_y)[:, 1:2], grads(v_z)[:, 2:3]
                w_xx, w_yy, w_zz = grads(w_x)[:, 0:1], grads(w_y)[:, 1:2], grads(w_z)[:, 2:3]

                inv_Re = 1.0 / self.Re
                f_e = u_x + v_y + w_z
                f_u = u * u_x + v * u_y + w * u_z + p_x - inv_Re * (u_xx + u_yy + u_zz)
                f_v = u * v_x + v * v_y + w * v_z + p_y - inv_Re * (v_xx + v_yy + v_zz)
                f_w = u * w_x + v * w_y + w * w_z + p_z - inv_Re * (w_xx + w_yy + w_zz)

                return uvwp, f_u, f_v, f_w, f_e


        # =============================================================================
        # 4. Graph Architecture & Knowledge Network
        # =============================================================================
        class ManifoldGNN(nn.Module):
            def __init__(self, input_dim=150, hidden_dim=128, output_dim=64, heads=4):
                super().__init__()
                self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
                self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

            def forward(self, x, edge_index, edge_weight=None):
                x = F.leaky_relu(self.gat1(x, edge_index, edge_weight), negative_slope=0.2)
                return F.leaky_relu(self.gat2(x, edge_index, edge_weight), negative_slope=0.2)


        class AdvancedKnowledgeNetwork:
            def __init__(self, device, k_neighbors=5, output_dim=64):
                self.device = device
                self.k_neighbors = k_neighbors
                self.gnn = ManifoldGNN(output_dim=output_dim).to(device)
                self.task_embeddings = None

            def build_and_train_graph(self, features, epochs=200):
                X = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)
                norms = torch.norm(X, p=2, dim=1, keepdim=True)
                sim_matrix = torch.mm(X / (norms + 1e-8), (X / (norms + 1e-8)).t())

                actual_k = min(self.k_neighbors + 1, len(X))
                values, indices = torch.topk(sim_matrix, k=actual_k, dim=1)

                edge_src, edge_dst, edge_w = [], [], []
                for i in range(len(X)):
                    for j_idx, sim_val in zip(indices[i], values[i]):
                        j = j_idx.item()
                        if i != j:
                            edge_src.append(i), edge_dst.append(j), edge_w.append(sim_val.item())

                self.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(self.device)
                self.edge_weights = torch.tensor(edge_w, dtype=torch.float32).to(self.device)
                self.dense_adj = torch.zeros((len(X), len(X)), device=self.device)
                self.dense_adj[self.edge_index[0], self.edge_index[1]] = self.edge_weights

                optimizer = optim.Adam(self.gnn.parameters(), lr=0.01)
                for _ in range(epochs):
                    optimizer.zero_grad()
                    z = self.gnn(X, self.edge_index, self.edge_weights)
                    z_norm = F.normalize(z, p=2, dim=1)
                    loss = F.mse_loss(torch.matmul(z_norm, z_norm.t()), self.dense_adj)
                    loss.backward()
                    optimizer.step()

                self.gnn.eval()
                with torch.no_grad():
                    self.task_embeddings = self.gnn(X, self.edge_index, self.edge_weights)
                return self.task_embeddings

            def find_anchor_task(self, alpha=0.65):
                degree = torch.zeros(len(self.task_embeddings), device=self.device)
                degree.index_add_(0, self.edge_index[0], torch.ones_like(self.edge_index[0], dtype=torch.float))
                deg_cen = degree / (degree.max() + 1e-8)

                eig_cen = torch.ones(len(self.task_embeddings), device=self.device)
                for _ in range(20):
                    eig_cen = torch.matmul(self.dense_adj, eig_cen)
                    eig_cen = eig_cen / (torch.norm(eig_cen) + 1e-8)

                return torch.argmax(alpha * eig_cen + (1 - alpha) * deg_cen).item()

            def select_next_task(self, trained_indices, remaining_indices):
                influence = torch.zeros(len(self.task_embeddings), device=self.device)
                influence[trained_indices] = 1.0
                for _ in range(3): influence = torch.matmul(self.dense_adj, influence)
                best_idx = max(remaining_indices, key=lambda idx: influence[idx].item())
                return best_idx, influence[best_idx].item()

            def get_transfer_components(self, target_idx, trained_indices, tau=0.1):
                sims = F.cosine_similarity(self.task_embeddings[target_idx].unsqueeze(0),
                                           self.task_embeddings[trained_indices])
                _, top_k = torch.topk(sims, min(self.k_neighbors, len(trained_indices)))
                neighbors = [trained_indices[i] for i in top_k.cpu().numpy()]
                weights = F.softmax(
                    torch.matmul(self.task_embeddings[neighbors], self.task_embeddings[target_idx]) / tau, dim=0)
                return neighbors, weights.cpu().numpy()


        # =============================================================================
        # 5. Optimization & Transfer Engine
        # =============================================================================
        class TransferTuner:
            def __init__(self, device, lambda_b=50.0):
                self.device = device
                self.lambda_b = lambda_b

            def execute_transfer(self, target_model, neighbor_indices, weights, model_dir, task_map):
                target_state = target_model.state_dict()
                fused_adapter = {k: torch.zeros_like(v).to(self.device) for k, v in target_state.items() if
                                 'adapter' in k}

                nearest_idx, encoder_loaded = neighbor_indices[0], False

                for idx, graph_idx in enumerate(neighbor_indices):
                    path = os.path.join(model_dir, f"task_{task_map[graph_idx]}_migrated.pth")
                    if not os.path.exists(path): path = os.path.join(model_dir, f"task_{task_map[graph_idx]}_NS.pth")

                    if os.path.exists(path):
                        src_state = torch.load(path, map_location=self.device)
                        if graph_idx == nearest_idx and not encoder_loaded:
                            for k in target_state.keys():
                                if 'encoder' in k: target_state[k].copy_(src_state[k])
                            encoder_loaded = True

                        for k in fused_adapter.keys(): fused_adapter[k] += weights[idx] * src_state[k]

                target_model.load_state_dict(fused_adapter, strict=False)
                return target_model

            def train_pinn(self, model, data, h_graph, is_anchor=False):
                coarse_ratio = 1.0 if is_anchor else 0.3

                # Rigorous Topology-Preserving Subsampling replaces random choice
                coords = data[:, 0:3]
                fields = data[:, 3:7]
                sampled_coords, sampled_fields = topology_preserving_subsampling(coords, fields,
                                                                                 retain_ratio=coarse_ratio)

                X_phys = torch.tensor(coords, dtype=torch.float32, requires_grad=True).to(self.device)
                X_bound = torch.tensor(sampled_coords, dtype=torch.float32).to(self.device)
                Y_bound = torch.tensor(sampled_fields, dtype=torch.float32).to(self.device)

                h_graph_exp = h_graph.unsqueeze(0).repeat(X_phys.shape[0], 1)
                h_graph_bnd = h_graph.unsqueeze(0).repeat(X_bound.shape[0], 1)

                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                model.train()

                for _ in range(10000 if is_anchor else 4000):
                    optimizer.zero_grad()
                    loss_data = torch.mean((model(X_bound, h_graph_bnd) - Y_bound) ** 2)
                    _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_exp)
                    loss_pde = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2 + f_e ** 2)
                    loss = self.lambda_b * loss_data + loss_pde
                    loss.backward()
                    optimizer.step()

                lbfgs = LBFGS(model.parameters(), max_iter=2000, tolerance_grad=1e-5, line_search_fn='strong_wolfe')

                def closure():
                    lbfgs.zero_grad()
                    loss_data = torch.mean((model(X_bound, h_graph_bnd) - Y_bound) ** 2)
                    _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_exp)
                    loss_pde = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2 + f_e ** 2)
                    loss = self.lambda_b * loss_data + loss_pde
                    loss.backward()
                    return loss

                lbfgs.step(closure)
                return model


        # =============================================================================
        # 6. Main Execution Pipeline
        # =============================================================================
        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--data_dir", type=str, default="./data")
            parser.add_argument("--output_dir", type=str, default="./output_models")
            parser.add_argument("--alpha", type=float, default=0.65)
            args = parser.parse_args()

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.makedirs(args.output_dir, exist_ok=True)

            extractor = SpectralPhysicalExtractor()
            knowledge_net = AdvancedKnowledgeNetwork(DEVICE, k_neighbors=5)
            tuner = TransferTuner(DEVICE, lambda_b=50.0)

            task_features, task_data_map, task_id_map = [], {}, {}
            for idx, csv_path in enumerate(sorted(glob.glob(os.path.join(args.data_dir, "case_*.csv")))):
                case_id = int(os.path.basename(csv_path).replace("case_", "").replace(".csv", ""))
                df = pd.read_csv(csv_path)

                coords, phys_fields = df[['x', 'y', 'z']].values, df[['u', 'v', 'w', 'p']].values
                coords_norm = nondimensionalize_coordinates(coords)

                tensor_3d = natural_neighbor_surrogate_resampling(coords_norm, phys_fields, resolution=64)
                task_features.append(extractor.compute_features(tensor_3d))

                task_data_map[idx] = np.hstack([coords_norm, phys_fields])
                task_id_map[idx] = case_id

            embeddings = knowledge_net.build_and_train_graph(task_features)
            anchor_idx = knowledge_net.find_anchor_task(alpha=args.alpha)

            anchor_model = GD_PINNs_Net().to(DEVICE)
            anchor_model = tuner.train_pinn(anchor_model, task_data_map[anchor_idx], embeddings[anchor_idx].detach(),
                                            is_anchor=True)
            torch.save(anchor_model.state_dict(),
                       os.path.join(args.output_dir, f"task_{task_id_map[anchor_idx]}_NS.pth"))

            trained_indices = [anchor_idx]
            remaining_indices = [i for i in range(len(task_features)) if i != anchor_idx]

            while remaining_indices:
                target_idx, _ = knowledge_net.select_next_task(trained_indices, remaining_indices)
                real_id = task_id_map[target_idx]

                neighbor_indices, weights = knowledge_net.get_transfer_components(target_idx, trained_indices)

                target_model = GD_PINNs_Net().to(DEVICE)
                target_model = tuner.execute_transfer(target_model, neighbor_indices, weights, args.output_dir,
                                                      task_id_map)
                target_model = tuner.train_pinn(target_model, task_data_map[target_idx],
                                                embeddings[target_idx].detach(), is_anchor=False)

                torch.save(target_model.state_dict(), os.path.join(args.output_dir, f"task_{real_id}_migrated.pth"))
                trained_indices.append(target_idx)
                remaining_indices.remove(target_idx)
        torch.save(target_model.state_dict(), os.path.join(args.output_dir, f"task_{real_id}_migrated.pth"))
        trained_indices.append(target_idx)
        remaining_indices.remove(target_idx)

    print("GD-PINNs Manifold Transfer Pipeline Completed Successfully.")
