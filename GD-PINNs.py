"""
GD-PINNs: Graph-Diffusion Physics-Informed Neural Networks
Complete implementation strictly following the paper methodology.
"""

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
from scipy.interpolate import LinearNDInterpolator
from scipy.fft import fftn, fftshift
from torch_geometric.nn import GATConv


# =============================================================================
# Stage 1: Spectral-Physical Manifold Parameterization
# =============================================================================

def nondimensionalize_coordinates(coords):
    """
    Eq. 6: x* = x / x_ref, normalized by characteristic length.
    Includes translation to bound the domain for PINN stability.

    Parameters
    ----------
    coords : np.ndarray
        Raw coordinates of shape (N_points, 3).

    Returns
    -------
    coords_norm : np.ndarray
        Normalized coordinates.
    l_ref : float
        Characteristic length scale.
    """
    coords_min = np.min(coords, axis=0)
    l_ref = np.max(np.max(coords, axis=0) - coords_min)
    coords_norm = (coords - coords_min) / l_ref
    return coords_norm, l_ref


def natural_neighbor_interpolation(coords_norm, fields, resolution=64):
    """
    Natural-neighbor interpolation onto unified Cartesian grid.
    Uses Delaunay triangulation for topology-preserving reconstruction.
    Preserves dominant low-frequency structures without spurious oscillations.

    Parameters
    ----------
    coords_norm : np.ndarray
        Normalized coordinates of shape (N_points, 3).
    fields : np.ndarray
        Physical fields of shape (N_points, C).
    resolution : int
        Grid resolution along each dimension.

    Returns
    -------
    np.ndarray
        Structured Cartesian tensor of shape (res, res, res, C).
    """
    axis = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing='ij')
    target_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    num_channels = fields.shape[1]
    structured = np.zeros((target_points.shape[0], num_channels), dtype=np.float64)

    for c in range(num_channels):
        interp = LinearNDInterpolator(coords_norm, fields[:, c])
        vals = interp(target_points)

        # Handle points outside convex hull with nearest-neighbor
        nan_mask = np.isnan(vals)
        if np.any(nan_mask):
            tree = cKDTree(coords_norm)
            _, nearest = tree.query(target_points[nan_mask], k=1)
            vals[nan_mask] = fields[nearest, c]

        structured[:, c] = vals

    return structured.reshape(resolution, resolution, resolution, num_channels)


class SpectralPhysicalExtractor:
    """
    Constructs compact manifold coordinates from three complementary descriptors:
    - FFT (80-dim): dominant global transport organization
    - PSD (60-dim): multiscale spectral-energy distribution
    - POD (10-dim): coherent physical deformation structures

    Total: 150-dimensional spectral-physical embedding.
    """

    def __init__(self, d_fft=80, d_psd=60, d_pod=10):
        self.d_fft = d_fft
        self.d_psd = d_psd
        self.d_pod = d_pod

    def extract_fft(self, tensor_3d):
        """
        Extract low-wavenumber FFT coefficients for global transport characterization.
        Retains 5x5x5 low-frequency cube (wavenumbers -2 to 2).
        """
        fft_features = []
        center = np.array(tensor_3d.shape[:3]) // 2

        for c in range(tensor_3d.shape[-1]):  # u, v, w, p
            fft_3d = np.abs(fftn(tensor_3d[..., c]))
            fft_3d = fftshift(fft_3d)

            # Extract 5x5x5 low-frequency cube
            low_freq = fft_3d[center[0]-2:center[0]+3,
                             center[1]-2:center[1]+3,
                             center[2]-2:center[2]+3]
            sorted_modes = np.sort(low_freq.flatten())[::-1][:20]
            sorted_modes = np.pad(sorted_modes, (0, max(0, 20 - len(sorted_modes))))
            fft_features.extend(sorted_modes / (np.sum(sorted_modes) + 1e-8))

        return np.array(fft_features[:self.d_fft])

    def extract_psd(self, tensor_3d):
        """
        Extract PSD for multiscale energy distribution.
        Computes along each axis for isotropic spectral energy characterization.
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
        Extract POD modes for coherent structure characterization.
        Retains modes preserving >95% cumulative energy.
        """
        snapshot = tensor_3d.reshape(-1, tensor_3d.shape[-1])

        try:
            _, S, _ = np.linalg.svd(snapshot, full_matrices=False)
            energy = S**2
            total_energy = np.sum(energy)

            # Retain modes preserving >95% cumulative energy
            cumsum = np.cumsum(energy) / total_energy
            k_95 = np.argmax(cumsum >= 0.95) + 1
            k_opt = min(k_95, self.d_pod)

            pod_features = energy[:k_opt] / total_energy
        except np.linalg.LinAlgError:
            pod_features = np.ones(self.d_pod) / self.d_pod

        return np.pad(pod_features, (0, max(0, self.d_pod - len(pod_features))))

    def compute_features(self, tensor_3d):
        """
        Compute combined spectral-physical embedding:
        z_i = [z_i^FFT, z_i^PSD, z_i^POD]
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
    Graph Attention Network for manifold diffusion.
    Implements anisotropic diffusion on the discrete solution manifold.
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
    Builds k-NN graph on the solution manifold using cosine similarity.
    Implements adaptive Anchor Support Set expansion with convergence criterion.
    """

    def __init__(self, device, k_neighbors=5, output_dim=64):
        self.device = device
        self.k_neighbors = k_neighbors
        self.output_dim = output_dim
        self.embeddings = None
        self.dense_adj = None

    def build_and_train_graph(self, features, epochs=200):
        """
        Build similarity graph and train GAT for manifold representation.

        Parameters
        ----------
        features : np.ndarray
            Spectral-physical embeddings of shape (N_tasks, 150).
        epochs : int
            Number of GAT training epochs.

        Returns
        -------
        torch.Tensor
            Graph-enhanced manifold embeddings of shape (N_tasks, 64).
        """
        X = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)

        # Cosine similarity matrix (Eq. 12)
        norms = torch.norm(X, p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(X / (norms + 1e-8), (X / (norms + 1e-8)).t())

        # k-NN selection
        actual_k = min(self.k_neighbors + 1, len(X))
        values, indices = torch.topk(sim_matrix, k=actual_k, dim=1)

        # Build sparse adjacency
        src, dst, w = [], [], []
        for i in range(len(X)):
            for j_idx, sim_val in zip(indices[i], values[i]):
                j = j_idx.item()
                if i != j:
                    src.append(i)
                    dst.append(j)
                    w.append(sim_val.item())

        edge_index = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        edge_weights = torch.tensor(w, dtype=torch.float32).to(self.device)

        # Dense adjacency for centrality computation
        self.dense_adj = torch.zeros((len(X), len(X)), device=self.device)
        self.dense_adj[edge_index[0], edge_index[1]] = edge_weights

        # Train GAT
        gnn = ManifoldGNN(input_dim=X.shape[1], output_dim=self.output_dim).to(self.device)
        optimizer = optim.Adam(gnn.parameters(), lr=0.01)

        for _ in range(epochs):
            optimizer.zero_grad()
            z = gnn(X, edge_index, edge_weights)
            z_norm = F.normalize(z, p=2, dim=1)

            # Reconstruction loss: preserve similarity structure
            loss = F.mse_loss(torch.matmul(z_norm, z_norm.t()), self.dense_adj)
            loss.backward()
            optimizer.step()

        gnn.eval()
        with torch.no_grad():
            self.embeddings = gnn(X, edge_index, edge_weights)

        return self.embeddings

    def expand_anchor_set(self, features, max_iter=100, tau_anchor=0.01):
        """
        Progressive anchor expansion until manifold geometry converges (Eq. 14).

        Parameters
        ----------
        features : np.ndarray
            Spectral-physical embeddings.
        max_iter : int
            Maximum expansion iterations.
        tau_anchor : float
            Convergence threshold for manifold geometry.

        Returns
        -------
        np.ndarray
            Indices of converged anchor tasks.
        """
        prev_cov = None

        for m in range(1, max_iter + 1):
            n_anchor = min(m * 10, len(features))

            if n_anchor >= len(features):
                return np.arange(len(features))

            anchor_indices = np.random.choice(len(features), n_anchor, replace=False)
            anchor_emb = self.embeddings[anchor_indices]

            # Compute covariance matrix of anchor embeddings
            cov = torch.cov(anchor_emb.T)

            if prev_cov is not None:
                diff_norm = torch.norm(cov - prev_cov, p='fro')
                prev_norm = torch.norm(prev_cov, p='fro') + 1e-8
                eta = diff_norm / prev_norm

                if eta.item() < tau_anchor:
                    print(f"[Anchor] Converged at step {m} with eta={eta.item():.4f}")
                    return anchor_indices

            prev_cov = cov

        return np.arange(len(features))

    def select_next_task(self, trained_indices, remaining_indices):
        """
        Select remaining tasks following graph-guided proximity.
        Implements graph diffusion for topological radiation evaluation.

        Parameters
        ----------
        trained_indices : list
            Indices of already trained tasks.
        remaining_indices : list
            Indices of remaining tasks.

        Returns
        -------
        int
            Index of the next task to train.
        """
        influence = torch.zeros(len(self.embeddings), device=self.device)
        influence[trained_indices] = 1.0

        # 3-step graph diffusion (Section 4.2)
        for _ in range(3):
            influence = torch.matmul(self.dense_adj, influence)

        best_idx = max(remaining_indices, key=lambda idx: influence[idx].item())
        return best_idx

    def get_transfer_weights(self, target_idx, trained_indices, tau=0.1):
        """
        Compute graph-weighted interpolation coefficients (Eq. 22).

        Parameters
        ----------
        target_idx : int
            Index of target task.
        trained_indices : list
            Indices of trained tasks.
        tau : float
            Temperature for softmax concentration.

        Returns
        -------
        nearest_idx : int
            Index of nearest neighbor for encoder inheritance.
        neighbor_idx : list
            Indices of k-nearest neighbors.
        weights : np.ndarray
            Interpolation weights for adapter fusion.
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

        # 1-NN for strict operator inheritance (Section 4.3)
        nearest_idx = neighbor_idx[0]

        return nearest_idx, neighbor_idx, weights


# =============================================================================
# Stage 3: Manifold-Conditioned Operator-Geometry Transfer
# =============================================================================

class GD_PINNs_Net(nn.Module):
    """
    Decoupled architecture with:
    - Locally Inherited Operator Encoder (invariant physics)
    - FiLM modulation (Eq. 21)
    - Geometric Manifold Adapter (geometry-dependent variations)

    Width: 100 neurons per hidden layer.
    """

    def __init__(self, Re=229.1451, graph_emb_dim=64):
        super().__init__()
        self.Re = Re

        # Locally Inherited Operator Encoder (invariant physics)
        self.encoder = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh()
        )

        # FiLM generator (Eq. 21): gamma and beta coefficients
        self.film_mlp = nn.Sequential(
            nn.Linear(graph_emb_dim, 100), nn.ReLU(),
            nn.Linear(100, 200)  # gamma (100) + beta (100)
        )

        # Geometric Manifold Adapter (geometry-dependent variations)
        self.adapter = nn.Sequential(
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 4)  # u, v, w, p
        )

    def forward(self, x, h_graph):
        """
        Forward pass with FiLM conditioning (Eq. 21).

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates of shape (N, 3).
        h_graph : torch.Tensor
            Graph-enhanced manifold embedding of shape (N, 64).

        Returns
        -------
        torch.Tensor
            Predicted fields (u, v, w, p) of shape (N, 4).
        """
        latent = self.encoder(x)
        film_params = self.film_mlp(h_graph)
        gamma = film_params[:, :100]
        beta = film_params[:, 100:]

        # Feature-wise Linear Modulation
        modulated = gamma * latent + beta
        return self.adapter(modulated)

    def net_f_NS(self, x, h_graph):
        """
        Navier-Stokes residual computation for physics-informed training.
        Computes continuity and momentum residuals (Eq. 7).

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates of shape (N, 3).
        h_graph : torch.Tensor
            Graph-enhanced manifold embedding.

        Returns
        -------
        tuple
            (uvwp, f_u, f_v, f_w, f_e) where f_* are PDE residuals.
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

        u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
        v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
        w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
        p_x, p_y, p_z = p_grads[:, 0:1], p_grads[:, 1:2], p_grads[:, 2:3]

        # Second derivatives
        u_xx = grads(u_x)[:, 0:1]
        u_yy = grads(u_y)[:, 1:2]
        u_zz = grads(u_z)[:, 2:3]
        v_xx = grads(v_x)[:, 0:1]
        v_yy = grads(v_y)[:, 1:2]
        v_zz = grads(v_z)[:, 2:3]
        w_xx = grads(w_x)[:, 0:1]
        w_yy = grads(w_y)[:, 1:2]
        w_zz = grads(w_z)[:, 2:3]

        inv_Re = 1.0 / self.Re

        # Continuity residual
        f_e = u_x + v_y + w_z

        # Momentum residuals (Eq. 7)
        f_u = u * u_x + v * u_y + w * u_z + p_x - inv_Re * (u_xx + u_yy + u_zz)
        f_v = u * v_x + v * v_y + w * v_z + p_y - inv_Re * (v_xx + v_yy + v_zz)
        f_w = u * w_x + v * w_y + w * w_z + p_z - inv_Re * (w_xx + w_yy + w_zz)

        return uvwp, f_u, f_v, f_w, f_e


class OperatorGeometryTransfer:
    """
    Manifold-conditioned transfer with decoupled operator inheritance.
    Implements 1-NN encoder inheritance and graph-weighted adapter interpolation.
    """

    def __init__(self, device, lambda_b=50.0):
        self.device = device
        self.lambda_b = lambda_b

    def hot_start_target_model(self, target_model, source_state_1nn, neighbor_states, weights):
        """
        Decoupled hot-start strategy:
        1. Exact inheritance of Encoder from nearest topological neighbor (1-NN).
        2. Graph-weighted structural fusion for Adapter (Eq. 22).

        Parameters
        ----------
        target_model : nn.Module
            Target model to initialize.
        source_state_1nn : dict
            State dict of nearest neighbor for encoder inheritance.
        neighbor_states : list
            State dicts of k-nearest neighbors for adapter fusion.
        weights : np.ndarray
            Interpolation weights for adapter fusion.

        Returns
        -------
        nn.Module
            Initialized target model.
        """
        target_state = target_model.state_dict()

        for k in target_state.keys():
            if 'encoder' in k and k in source_state_1nn:
                # Strict 1-NN inheritance (Section 4.3)
                target_state[k].copy_(source_state_1nn[k])
            elif 'adapter' in k:
                # Graph-weighted interpolation (Eq. 22)
                fused = torch.zeros_like(target_state[k])
                for w, state in zip(weights, neighbor_states):
                    if k in state:
                        fused += w * state[k]
                target_state[k].copy_(fused)

        target_model.load_state_dict(target_state, strict=False)
        return target_model

    def topology_preserving_subsample(self, coords, fields, retain_ratio=0.3):
        """
        Uniform spatial downsampling via voxel grid filtering (Section 4.4).
        Replaces random sampling to preserve topological structures.

        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates of shape (N, 3).
        fields : np.ndarray
            Physical fields of shape (N, 4).
        retain_ratio : float
            Fraction of points to retain (30% for transfer tasks).

        Returns
        -------
        tuple
            (coords_sampled, fields_sampled)
        """
        num_points = len(coords)
        target_points = int(num_points * retain_ratio)

        if target_points <= 0 or target_points >= num_points:
            return coords, fields

        # Voxel size calculation
        extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        volume = np.prod(np.where(extent > 0, extent, 1e-8))
        voxel_size = (volume / target_points) ** (1.0 / 3.0)

        # Assign points to voxels
        voxel_idx = np.floor((coords - np.min(coords, axis=0)) / voxel_size).astype(int)
        hashes = voxel_idx[:, 0] + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2] * 1000000

        # Sample one point per voxel
        df = pd.DataFrame({'hash': hashes, 'idx': np.arange(num_points)})
        sampled_idx = df.groupby('hash').first()['idx'].values

        # Handle undersampling
        if len(sampled_idx) < target_points:
            remaining = target_points - len(sampled_idx)
            available = np.setdiff1d(np.arange(num_points), sampled_idx)
            topup = np.random.choice(available, remaining, replace=False)
            sampled_idx = np.concatenate([sampled_idx, topup])

        return coords[sampled_idx], fields[sampled_idx]

    def train_pinn(self, model, coords, fields, h_graph, is_anchor=False, coarse_ratio=0.3):
        """
        Hybrid optimization with topology-preserving coarse-grid conditioning (Section 4.4).

        Parameters
        ----------
        model : nn.Module
            PINN model to train.
        coords : np.ndarray
            Full spatial coordinates (physics collocation points).
        fields : np.ndarray
            Full physical fields (for supervision).
        h_graph : torch.Tensor
            Graph-enhanced manifold embedding.
        is_anchor : bool
            Whether this is an anchor task (uses 100% data).
        coarse_ratio : float
            Coarse-grid retention ratio (30% for transfer).

        Returns
        -------
        nn.Module
            Trained model.
        """
        # Topology-preserving subsampling (Section 4.4)
        if not is_anchor:
            coords_sampled, fields_sampled = self.topology_preserving_subsample(
                coords, fields, coarse_ratio
            )
        else:
            coords_sampled, fields_sampled = coords, fields

        # Physics collocation points (full domain)
        X_phys = torch.tensor(coords, dtype=torch.float32, requires_grad=True).to(self.device)

        # Boundary/supervision points (coarse-grid)
        X_bound = torch.tensor(coords_sampled, dtype=torch.float32).to(self.device)
        Y_bound = torch.tensor(fields_sampled, dtype=torch.float32).to(self.device)

        h_graph_phys = h_graph.unsqueeze(0).repeat(X_phys.shape[0], 1)
        h_graph_bound = h_graph.unsqueeze(0).repeat(X_bound.shape[0], 1)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        # Adam phase
        n_epochs = 10000 if is_anchor else 4000
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Data loss on coarse-grid observations
            pred_bound = model(X_bound, h_graph_bound)
            loss_data = torch.mean((pred_bound - Y_bound) ** 2)

            # PDE residual loss (Eq. 24)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_phys)
            loss_pde = torch.mean(f_u**2 + f_v**2 + f_w**2 + f_e**2)

            # Combined loss (Eq. 24)
            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"    Epoch {epoch+1}: Loss={loss.item():.6e}")

        # L-BFGS-B refinement
        print("    L-BFGS-B refinement...")
        lbfgs = LBFGS(model.parameters(), max_iter=2000,
                      tolerance_grad=1e-5, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            pred_bound = model(X_bound, h_graph_bound)
            loss_data = torch.mean((pred_bound - Y_bound) ** 2)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys, h_graph_phys)
            loss_pde = torch.mean(f_u**2 + f_v**2 + f_w**2 + f_e**2)
            loss = self.lambda_b * loss_data + loss_pde
            loss.backward()
            return loss

        lbfgs.step(closure)

        return model


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GD-PINNs: Manifold-Guided Transfer Learning')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing case_*.csv files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save models')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Cartesian grid resolution')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='Number of graph neighbors')
    parser.add_argument('--lambda_b', type=float, default=50.0,
                        help='Boundary loss weight (Eq. 24)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Initialize components
    extractor = SpectralPhysicalExtractor()
    graph_constructor = GraphManifoldConstructor(device, k_neighbors=args.k_neighbors)
    transfer_engine = OperatorGeometryTransfer(device, lambda_b=args.lambda_b)

    # ========================================================================
    # Stage 1: Spectral-Physical Embedding (Section 4.1)
    # ========================================================================
    print("\n" + "="*60)
    print("Stage 1: Spectral-Physical Manifold Parameterization")
    print("="*60)

    features = []
    task_data = {}
    task_tensors = {}
    task_ids = {}

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, 'case_*.csv')))
    print(f"Found {len(csv_files)} task files")

    for idx, csv_path in enumerate(csv_files):
        task_id = int(os.path.basename(csv_path).replace('case_', '').replace('.csv', ''))
        print(f"  Processing task {task_id}...")

        df = pd.read_csv(csv_path)
        coords = df[['x', 'y', 'z']].values
        fields = df[['u', 'v', 'w', 'p']].values

        # Nondimensionalization (Eq. 6)
        coords_norm, _ = nondimensionalize_coordinates(coords)

        # Natural-neighbor interpolation onto Cartesian grid
        tensor_3d = natural_neighbor_interpolation(coords_norm, fields, args.resolution)

        # Spectral-physical embedding
        z_i = extractor.compute_features(tensor_3d)
        features.append(z_i)

        # Store for PINN training
        task_data[idx] = np.hstack([coords_norm, fields])
        task_tensors[idx] = (coords_norm, fields)
        task_ids[idx] = task_id

    features = np.array(features)
    print(f"  Extracted {len(features)} embeddings of dimension {features.shape[1]}")

    # ========================================================================
    # Stage 2: Adaptive Graph Initialization (Section 4.2)
    # ========================================================================
    print("\n" + "="*60)
    print("Stage 2: Adaptive Graph Initialization")
    print("="*60)

    print("  Building and training manifold graph...")
    embeddings = graph_constructor.build_and_train_graph(features)

    print("  Expanding anchor support set...")
    anchor_indices = graph_constructor.expand_anchor_set(features)
    anchor_task_ids = [task_ids[i] for i in anchor_indices]
    print(f"  Converged Anchor Support Set: {anchor_task_ids} (size: {len(anchor_indices)})")

    # ========================================================================
    # Stage 3: Anchor Training
    # ========================================================================
    print("\n" + "="*60)
    print("Stage 3: Baseline Training for Anchor Support Set")
    print("="*60)

    for anchor_idx in anchor_indices:
        print(f"\n  Training Anchor Task: {task_ids[anchor_idx]}")
        anchor_model = GD_PINNs_Net().to(device)
        h_anchor = embeddings[anchor_idx].detach()

        coords, fields = task_tensors[anchor_idx]
        anchor_model = transfer_engine.train_pinn(
            anchor_model, coords, fields, h_anchor, is_anchor=True
        )
        torch.save(anchor_model.state_dict(),
                   os.path.join(args.output_dir, f'task_{task_ids[anchor_idx]}_anchor.pth'))

    # ========================================================================
    # Stage 4: Progressive Transfer (Section 4.3)
    # ========================================================================
    print("\n" + "="*60)
    print("Stage 4: Operator-Geometry Decoupled Progressive Transfer")
    print("="*60)

    trained_indices = list(anchor_indices)
    remaining_indices = [i for i in range(len(features)) if i not in trained_indices]

    while remaining_indices:
        # Select next task via graph-guided proximity
        target_idx = graph_constructor.select_next_task(trained_indices, remaining_indices)
        task_id = task_ids[target_idx]
        print(f"\n  Migrating to target task {task_id} (Graph-guided proximity)")

        # Get transfer components
        nearest_idx, neighbor_idx, weights = graph_constructor.get_transfer_weights(
            target_idx, trained_indices
        )
        print(f"    Nearest neighbor: {task_ids[nearest_idx]}")
        print(f"    K-nearest neighbors: {[task_ids[i] for i in neighbor_idx]}")

        # Load neighbor states
        neighbor_states = []
        for ni in neighbor_idx:
            nid = task_ids[ni]
            migrated_path = os.path.join(args.output_dir, f'task_{nid}_migrated.pth')
            anchor_path = os.path.join(args.output_dir, f'task_{nid}_anchor.pth')
            load_path = migrated_path if os.path.exists(migrated_path) else anchor_path
            neighbor_states.append(torch.load(load_path, map_location=device))

        state_1nn = neighbor_states[neighbor_idx.index(nearest_idx)]

        # Initialize target model with decoupled hot-start
        target_model = GD_PINNs_Net().to(device)
        target_model = transfer_engine.hot_start_target_model(
            target_model, state_1nn, neighbor_states, weights
        )

        # Train with coarse-grid conditioning (30% retention)
        h_target = embeddings[target_idx].detach()
        coords, fields = task_tensors[target_idx]
        target_model = transfer_engine.train_pinn(
            target_model, coords, fields, h_target,
            is_anchor=False, coarse_ratio=0.3
        )

        # Save migrated model
        torch.save(target_model.state_dict(),
                   os.path.join(args.output_dir, f'task_{task_id}_migrated.pth'))

        trained_indices.append(target_idx)
        remaining_indices.remove(target_idx)

        print(f"  Completed task {task_id} (remaining: {len(remaining_indices)})")

    # ========================================================================
    # Completion
    # ========================================================================
    print("\n" + "="*60)
    print("GD-PINNs Pipeline Completed Successfully!")
    print("="*60)
    print(f"All models saved to: {args.output_dir}")
    print(f"Total tasks processed: {len(trained_indices)}")
    print(f"Anchor tasks: {len(anchor_indices)}")
    print(f"Transfer tasks: {len(trained_indices) - len(anchor_indices)}")


if __name__ == '__main__':
    main()
