from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.fft import fftn, fftshift
from scipy.spatial import cKDTree
from torch.optim import LBFGS


EPS = 1.0e-12


# =============================================================================
# Reproducibility and task I/O
# =============================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TaskRecord:
    index: int
    task_id: int
    csv_path: str


def discover_tasks(data_dir: str) -> List[TaskRecord]:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "case_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No case_*.csv files found under: {data_dir}")

    tasks: List[TaskRecord] = []
    for idx, path in enumerate(csv_files):
        stem = Path(path).stem
        try:
            task_id = int(stem.replace("case_", ""))
        except ValueError as exc:
            raise ValueError(f"Cannot parse integer task id from {path}") from exc
        tasks.append(TaskRecord(idx, task_id, path))
    return tasks


def infer_field_columns(csv_path: str) -> List[str]:
    cols = list(pd.read_csv(csv_path, nrows=2).columns)
    required = ["x", "y", "z", "u", "v", "w", "p"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    field_cols = ["u", "v", "w", "p"]
    if "c" in cols:
        field_cols.append("c")
    return field_cols


def load_task_csv(
    task: TaskRecord,
    field_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(task.csv_path)
    required = ["x", "y", "z", *field_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Task {task.task_id} missing columns: {missing}")

    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
    fields = df[list(field_cols)].to_numpy(dtype=np.float64, copy=True)

    boundary_mask = None
    for name in ("is_boundary", "boundary"):
        if name in df.columns:
            boundary_mask = df[name].to_numpy().astype(bool)
            break
    return coords, fields, boundary_mask


# =============================================================================
# Stage 1: reference coordinates and spectral-physical descriptors
# =============================================================================


def nondimensionalize_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Map coordinates to the manuscript reference form

        X = 2 (x - x_min) / L_ref - 1,

    with L_ref taken as the maximum Cartesian extent. The physical chain-rule factor is

        d/dx = (2/L_ref) d/dX.

    Returns
    -------
    coords_ref : (N, 3)
    l_ref      : characteristic length
    coords_min : per-axis translation used for the map
    """
    coords = np.asarray(coords, dtype=np.float64)
    coords_min = coords.min(axis=0)
    extents = coords.max(axis=0) - coords_min
    l_ref = float(np.max(extents))
    if not np.isfinite(l_ref) or l_ref <= 0.0:
        raise ValueError("Degenerate geometry: characteristic length must be positive.")
    coords_ref = 2.0 * (coords - coords_min) / l_ref - 1.0
    return coords_ref, l_ref, coords_min


def make_cartesian_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    if resolution < 4:
        raise ValueError("Cartesian resolution must be >= 4.")
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([gx, gy, gz], axis=-1)
    return grid, grid.reshape(-1, 3)


def reconstruct_cartesian_idw(
    coords_ref: np.ndarray,
    fields: np.ndarray,
    resolution: int,
    k_neighbors: int = 8,
    power: float = 2.0,
    chunk_size: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Portable local reconstruction from an unstructured point cloud to a unified Cartesian
    grid using inverse-distance weighting. Exact matches are copied without weighting.

    This function intentionally does *not* call itself natural-neighbor interpolation.
    """
    coords_ref = np.asarray(coords_ref, dtype=np.float64)
    fields = np.asarray(fields, dtype=np.float64)
    if coords_ref.ndim != 2 or coords_ref.shape[1] != 3:
        raise ValueError("coords_ref must have shape (N, 3).")
    if fields.ndim != 2 or fields.shape[0] != coords_ref.shape[0]:
        raise ValueError("fields must have shape (N, C) and match coords_ref.")

    grid_4d, targets = make_cartesian_grid(resolution)
    tree = cKDTree(coords_ref)
    k = max(1, min(int(k_neighbors), len(coords_ref)))
    out = np.empty((len(targets), fields.shape[1]), dtype=np.float64)

    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        q = targets[start:stop]
        try:
            dists, inds = tree.query(q, k=k, workers=-1)
        except TypeError:  # older SciPy
            dists, inds = tree.query(q, k=k)

        if k == 1:
            dists = dists[:, None]
            inds = inds[:, None]

        exact = dists[:, 0] <= 1.0e-14
        vals = np.empty((len(q), fields.shape[1]), dtype=np.float64)
        if np.any(exact):
            vals[exact] = fields[inds[exact, 0]]

        non = ~exact
        if np.any(non):
            dn = np.maximum(dists[non], 1.0e-14)
            w = 1.0 / np.power(dn, power)
            w /= np.sum(w, axis=1, keepdims=True)
            vals[non] = np.sum(fields[inds[non]] * w[..., None], axis=1)
        out[start:stop] = vals

    tensor = out.reshape(resolution, resolution, resolution, fields.shape[1])
    return grid_4d, tensor


def coarse_resolution_from_ratio(full_resolution: int, retain_ratio: float) -> int:
    """
    Choose a regular coarse Cartesian resolution whose point count is approximately the
    requested 3-D retention ratio. Example: 64 -> 43 gives 43^3 / 64^3 ~= 0.303.
    """
    if not 0.0 < retain_ratio <= 1.0:
        raise ValueError("retain_ratio must lie in (0, 1].")
    if retain_ratio >= 1.0:
        return full_resolution
    coarse = int(round(full_resolution * retain_ratio ** (1.0 / 3.0)))
    return max(4, min(full_resolution, coarse))


class SpectralPhysicalExtractor:
    """
    150-dimensional descriptor:
      FFT: 80  = 20 fixed low-wavenumber amplitudes x 4 variables (u,v,w,p)
      PSD: 60  = 20 radial shell energies x 3 velocity components (u,v,w)
      POD: 10  = leading normalized modal energies, zero-padded as needed

    Fixed FFT locations preserve directional information; amplitudes remove translation
    phase dependence. Radial PSD shell averaging is rotationally robust in the spatial
    frame. POD follows the manuscript's spatial-by-field snapshot matrix definition.
    """

    def __init__(self, d_fft: int = 80, d_psd: int = 60, d_pod: int = 10):
        if (d_fft, d_psd, d_pod) != (80, 60, 10):
            raise ValueError("This reference implementation fixes descriptor dimensions to 80/60/10.")
        self.d_fft = d_fft
        self.d_psd = d_psd
        self.d_pod = d_pod

    @staticmethod
    def _low_frequency_offsets(n_modes: int = 20) -> List[Tuple[int, int, int]]:
        candidates: List[Tuple[int, int, int]] = []
        radius = 0
        while len(candidates) < n_modes:
            radius += 1
            candidates = [
                (i, j, k)
                for i in range(-radius, radius + 1)
                for j in range(-radius, radius + 1)
                for k in range(-radius, radius + 1)
            ]
            # Deterministic order: low |k| first, then preserve Cartesian orientation.
            candidates.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2 + t[2] ** 2, t[0], t[1], t[2]))
        return candidates[:n_modes]

    def extract_fft(self, tensor_3d: np.ndarray) -> np.ndarray:
        if tensor_3d.shape[-1] < 4:
            raise ValueError("FFT descriptor requires at least u,v,w,p fields.")
        offsets = self._low_frequency_offsets(20)
        shape = np.array(tensor_3d.shape[:3], dtype=int)
        center = shape // 2

        features: List[float] = []
        for c in range(4):
            spec = fftshift(np.abs(fftn(tensor_3d[..., c])))
            vals = []
            for off in offsets:
                idx = center + np.asarray(off, dtype=int)
                if np.any(idx < 0) or np.any(idx >= shape):
                    vals.append(0.0)
                else:
                    vals.append(float(spec[tuple(idx)]))
            vals = np.asarray(vals, dtype=np.float64)
            vals /= np.sum(vals) + EPS
            features.extend(vals.tolist())
        return np.asarray(features, dtype=np.float64)

    @staticmethod
    def _radial_shell_psd(field: np.ndarray, n_shells: int = 20) -> np.ndarray:
        spec = np.abs(fftn(field)) ** 2
        nx, ny, nz = field.shape
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        radius = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        # Unit-width isotropic wavenumber shells, matching the shell-average interpretation.
        shell_id = np.floor(radius + 1.0e-9).astype(np.int64)
        psd = np.zeros(n_shells, dtype=np.float64)
        for s in range(n_shells):
            mask = shell_id == s
            if np.any(mask):
                psd[s] = float(np.mean(spec[mask]))
        psd /= np.sum(psd) + EPS
        return psd

    def extract_psd(self, tensor_3d: np.ndarray) -> np.ndarray:
        if tensor_3d.shape[-1] < 3:
            raise ValueError("PSD descriptor requires u,v,w fields.")
        return np.concatenate([self._radial_shell_psd(tensor_3d[..., c], 20) for c in range(3)])

    def extract_pod(self, tensor_3d: np.ndarray) -> np.ndarray:
        # Manuscript form A_i in R^{|X| x 4}. It can have at most four non-zero singular
        # values for the SWRO four-field descriptor; remaining POD slots are zero-padded.
        A = tensor_3d[..., :4].reshape(-1, 4)
        A = A - np.mean(A, axis=0, keepdims=True)
        try:
            S = np.linalg.svd(A, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.zeros(self.d_pod, dtype=np.float64)
        energy = S ** 2
        total = float(np.sum(energy))
        if total <= EPS:
            return np.zeros(self.d_pod, dtype=np.float64)
        energy = energy / total
        cumulative = np.cumsum(energy)
        k95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
        k = min(k95, self.d_pod, len(energy))
        out = np.zeros(self.d_pod, dtype=np.float64)
        out[:k] = energy[:k]
        return out

    def compute_features(self, tensor_3d: np.ndarray) -> np.ndarray:
        z = np.concatenate([
            self.extract_fft(tensor_3d),
            self.extract_psd(tensor_3d),
            self.extract_pod(tensor_3d),
        ])
        if z.shape != (150,):
            raise RuntimeError(f"Descriptor has unexpected shape {z.shape}, expected (150,).")
        return z


# =============================================================================
# Field standardization from the Anchor Support Set
# =============================================================================


class FieldStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std = np.maximum(self.std, 1.0e-8)

    @classmethod
    def fit_anchor_support(
        cls,
        tasks: Sequence[TaskRecord],
        anchor_indices: Sequence[int],
        field_cols: Sequence[str],
    ) -> "FieldStandardizer":
        count = 0
        sum_ = np.zeros(len(field_cols), dtype=np.float64)
        sumsq = np.zeros(len(field_cols), dtype=np.float64)
        for idx in anchor_indices:
            _, fields, _ = load_task_csv(tasks[idx], field_cols)
            count += fields.shape[0]
            sum_ += np.sum(fields, axis=0)
            sumsq += np.sum(fields ** 2, axis=0)
        if count == 0:
            raise ValueError("Cannot fit field standardizer with an empty anchor set.")
        mean = sum_ / count
        var = np.maximum(sumsq / count - mean ** 2, 1.0e-16)
        return cls(mean, np.sqrt(var))

    def transform(self, fields: np.ndarray) -> np.ndarray:
        return (fields - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


# =============================================================================
# Graph construction, GAT, anchor selection and diffusion
# =============================================================================


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, EPS)
    return np.clip(x @ x.T, -1.0, 1.0)


def sparse_knn_adjacency(features: np.ndarray, k_neighbors: int) -> np.ndarray:
    sim = cosine_similarity_matrix(features)
    n = len(sim)
    k = min(max(1, k_neighbors), max(1, n - 1))
    A = np.zeros_like(sim)
    for i in range(n):
        order = np.argsort(sim[i])[::-1]
        order = order[order != i][:k]
        A[i, order] = np.maximum(sim[i, order], 0.0)
    # Undirected union to represent local manifold connectivity.
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    return A


def power_eigenvector_centrality(A: np.ndarray, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    n = A.shape[0]
    x = np.ones(n, dtype=np.float64) / math.sqrt(max(1, n))
    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm <= EPS:
            return np.ones(n, dtype=np.float64) / max(1, n)
        x_new /= norm
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    x = np.abs(x)
    return x / (np.max(x) + EPS)


def composite_centrality_ranking(A: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("centrality alpha must be in [0,1].")
    A_wo_diag = A.copy()
    np.fill_diagonal(A_wo_diag, 0.0)
    degree = A_wo_diag.sum(axis=1)
    degree = degree / (np.max(degree) + EPS)
    eigen = power_eigenvector_centrality(A_wo_diag)
    score = alpha * eigen + (1.0 - alpha) * degree
    return np.argsort(score)[::-1]


class DenseGraphAttentionLayer(nn.Module):
    """Dense two-term GAT attention for ~10^3 task graphs; no torch-geometric dependency."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, concat: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.weight = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.edge_scale = nn.Parameter(torch.ones(heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))
        nn.init.ones_(self.edge_scale)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # h: H x N x D
        h = torch.einsum("ni,hio->hno", x, self.weight)
        src = torch.einsum("hnd,hd->hn", h, self.attn_src)
        dst = torch.einsum("hnd,hd->hn", h, self.attn_dst)
        e = src.unsqueeze(-1) + dst.unsqueeze(-2)

        # Cosine similarity acts as a prior; learned attention remains anisotropic.
        edge_prior = torch.log(adjacency.clamp_min(1.0e-8)).unsqueeze(0)
        e = e + self.edge_scale[:, None, None] * edge_prior
        e = F.leaky_relu(e, negative_slope=0.2)

        mask = adjacency > 0
        e = e.masked_fill(~mask.unsqueeze(0), -torch.inf)
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum("hij,hjd->hid", alpha, h)

        if self.concat:
            return out.permute(1, 0, 2).reshape(x.shape[0], self.heads * self.out_dim)
        return out.mean(dim=0)


class ManifoldGNN(nn.Module):
    def __init__(self, input_dim: int = 150, hidden_dim: int = 128, output_dim: int = 64, heads: int = 4):
        super().__init__()
        self.gat1 = DenseGraphAttentionLayer(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = DenseGraphAttentionLayer(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.gat1(x, adjacency), negative_slope=0.2)
        x = F.leaky_relu(self.gat2(x, adjacency), negative_slope=0.2)
        return x


class GraphManifoldConstructor:
    def __init__(
        self,
        device: torch.device,
        k_neighbors: int = 5,
        output_dim: int = 64,
        centrality_alpha: float = 0.65,
    ):
        self.device = device
        self.k_neighbors = k_neighbors
        self.output_dim = output_dim
        self.centrality_alpha = centrality_alpha
        self.adjacency_np: Optional[np.ndarray] = None
        self.adjacency: Optional[torch.Tensor] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.gnn: Optional[ManifoldGNN] = None

    def preliminary_ranking(self, coarse_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = sparse_knn_adjacency(coarse_features, self.k_neighbors)
        ranking = composite_centrality_ranking(A, self.centrality_alpha)
        return ranking, A

    @staticmethod
    def expand_anchor_set(
        ranking: Sequence[int],
        fine_feature_getter,
        tau_anchor: float = 0.01,
        batch_size: int = 10,
        min_batches: int = 2,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Nested, centrality-guided expansion. Fine features are requested lazily only for
        anchors that have entered the support set.
        """
        selected: List[int] = []
        fine_features: Dict[int, np.ndarray] = {}
        prev_cov: Optional[np.ndarray] = None

        ranking = list(map(int, ranking))
        for batch_no, start in enumerate(range(0, len(ranking), batch_size), start=1):
            new_indices = ranking[start:start + batch_size]
            for idx in new_indices:
                if idx not in fine_features:
                    fine_features[idx] = np.asarray(fine_feature_getter(idx), dtype=np.float64)
                selected.append(idx)

            Z = np.stack([fine_features[i] for i in selected], axis=0)
            if len(Z) >= 2:
                cov = np.cov(Z, rowvar=False)
            else:
                cov = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)

            if prev_cov is not None and batch_no >= min_batches:
                eta = np.linalg.norm(cov - prev_cov, ord="fro") / (np.linalg.norm(prev_cov, ord="fro") + EPS)
                print(f"  Anchor expansion: n={len(selected):4d}, eta={eta:.4e}")
                if eta < tau_anchor:
                    return np.asarray(selected, dtype=int), fine_features
            prev_cov = cov

        return np.asarray(selected, dtype=int), fine_features

    def build_and_train_graph(self, features: np.ndarray, epochs: int = 200, lr: float = 1.0e-3) -> torch.Tensor:
        self.adjacency_np = sparse_knn_adjacency(features, self.k_neighbors)
        X = torch.tensor(features, dtype=torch.float32, device=self.device)
        A = torch.tensor(self.adjacency_np, dtype=torch.float32, device=self.device)
        self.adjacency = A

        gnn = ManifoldGNN(input_dim=X.shape[1], output_dim=self.output_dim, heads=4).to(self.device)
        optimizer = optim.Adam(gnn.parameters(), lr=lr)

        off_diag = ~torch.eye(len(X), dtype=torch.bool, device=self.device)
        edge_mask = (A > 0) & off_diag
        weights = torch.ones_like(A)
        weights[edge_mask] = 5.0

        for epoch in range(max(0, epochs)):
            optimizer.zero_grad(set_to_none=True)
            z = gnn(X, A)
            z_norm = F.normalize(z, p=2, dim=1)
            pred = z_norm @ z_norm.T
            loss = (((pred - A) ** 2) * weights)[off_diag].mean()
            loss.backward()
            optimizer.step()
            if epochs > 0 and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  GAT epoch {epoch + 1:4d}/{epochs}: structural loss={loss.item():.4e}")

        gnn.eval()
        with torch.no_grad():
            self.embeddings = gnn(X, A)
        self.gnn = gnn
        return self.embeddings

    def select_next_task(self, trained_indices: Sequence[int], remaining_indices: Sequence[int], diffusion_steps: int = 3) -> int:
        if self.adjacency is None:
            raise RuntimeError("Graph has not been constructed.")
        influence = torch.zeros(self.adjacency.shape[0], device=self.device)
        influence[list(trained_indices)] = 1.0
        P = self.adjacency / (self.adjacency.sum(dim=1, keepdim=True) + 1.0e-8)
        for _ in range(diffusion_steps):
            influence = P @ influence
        return max(remaining_indices, key=lambda idx: float(influence[int(idx)].item()))

    def get_transfer_weights(
        self,
        target_idx: int,
        trained_indices: Sequence[int],
        tau: float,
    ) -> Tuple[int, List[int], np.ndarray]:
        if self.embeddings is None:
            raise RuntimeError("Graph embeddings are unavailable.")
        trained = list(map(int, trained_indices))
        target_emb = self.embeddings[target_idx]
        candidate_embs = self.embeddings[trained]
        sims = F.cosine_similarity(target_emb.unsqueeze(0), candidate_embs, dim=1)
        k = min(self.k_neighbors, len(trained))
        top_sims, top_pos = torch.topk(sims, k=k)
        neighbor_idx = [trained[int(i)] for i in top_pos.detach().cpu().numpy()]
        logits = top_sims / max(float(tau), 1.0e-6)
        weights = F.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64)
        nearest_idx = neighbor_idx[0]
        return nearest_idx, neighbor_idx, weights


# =============================================================================
# Stage 3: manifold-conditioned operator-geometry PINN
# =============================================================================


class GD_PINNs_Net(nn.Module):
    def __init__(
        self,
        output_dim: int,
        graph_emb_dim: int = 64,
        re_number: float = 229.1451,
        pe_number: Optional[float] = None,
    ):
        super().__init__()
        if output_dim not in (4, 5):
            raise ValueError("output_dim must be 4 (u,v,w,p) or 5 (+c).")
        self.output_dim = output_dim
        self.re_number = float(re_number)
        self.pe_number = None if pe_number is None else float(pe_number)

        # Four hidden layers x 100: Locally Inherited Operator Encoder.
        self.encoder = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
        )

        self.film_mlp = nn.Sequential(
            nn.Linear(graph_emb_dim, 100), nn.Tanh(),
            nn.Linear(100, 200),
        )

        # Three hidden layers x 100: Geometric Manifold Adapter.
        self.adapter = nn.Sequential(
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, output_dim),
        )

        self.register_buffer("field_mean", torch.zeros(output_dim, dtype=torch.float32))
        self.register_buffer("field_std", torch.ones(output_dim, dtype=torch.float32))

    def set_field_standardizer(self, standardizer: FieldStandardizer) -> None:
        if len(standardizer.mean) != self.output_dim:
            raise ValueError("Field standardizer dimension does not match network output.")
        self.field_mean.copy_(torch.as_tensor(standardizer.mean, dtype=self.field_mean.dtype, device=self.field_mean.device))
        self.field_std.copy_(torch.as_tensor(standardizer.std, dtype=self.field_std.dtype, device=self.field_std.device))

    def forward(self, x: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        film = self.film_mlp(h_graph)
        gamma, beta = film[:, :100], film[:, 100:]
        # Identity-centred scaling avoids destructive zero-centred FiLM at initialization.
        modulated = (1.0 + gamma) * latent + beta
        return self.adapter(modulated)  # standardized fields

    def forward_physical(self, x: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        pred_std = self.forward(x, h_graph)
        return pred_std * self.field_std.unsqueeze(0) + self.field_mean.unsqueeze(0)

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

    def physics_residuals(
        self,
        x_ref: torch.Tensor,
        h_graph: torch.Tensor,
        l_ref: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Keep a complete graph through second derivatives so viscous/diffusive terms
        # contribute correctly to parameter gradients.
        x = x_ref.detach().clone().requires_grad_(True)
        q = self.forward_physical(x, h_graph)
        u, v, w, p = [q[:, i:i + 1] for i in range(4)]

        gu, gv, gw, gp = [self._grad(y, x) for y in (u, v, w, p)]
        s1 = 2.0 / float(l_ref)
        s2 = s1 ** 2

        u_x, u_y, u_z = gu[:, 0:1] * s1, gu[:, 1:2] * s1, gu[:, 2:3] * s1
        v_x, v_y, v_z = gv[:, 0:1] * s1, gv[:, 1:2] * s1, gv[:, 2:3] * s1
        w_x, w_y, w_z = gw[:, 0:1] * s1, gw[:, 1:2] * s1, gw[:, 2:3] * s1
        p_x, p_y, p_z = gp[:, 0:1] * s1, gp[:, 1:2] * s1, gp[:, 2:3] * s1

        g_u_x = self._grad(gu[:, 0:1], x)
        g_u_y = self._grad(gu[:, 1:2], x)
        g_u_z = self._grad(gu[:, 2:3], x)
        g_v_x = self._grad(gv[:, 0:1], x)
        g_v_y = self._grad(gv[:, 1:2], x)
        g_v_z = self._grad(gv[:, 2:3], x)
        g_w_x = self._grad(gw[:, 0:1], x)
        g_w_y = self._grad(gw[:, 1:2], x)
        g_w_z = self._grad(gw[:, 2:3], x)

        u_xx, u_yy, u_zz = g_u_x[:, 0:1] * s2, g_u_y[:, 1:2] * s2, g_u_z[:, 2:3] * s2
        v_xx, v_yy, v_zz = g_v_x[:, 0:1] * s2, g_v_y[:, 1:2] * s2, g_v_z[:, 2:3] * s2
        w_xx, w_yy, w_zz = g_w_x[:, 0:1] * s2, g_w_y[:, 1:2] * s2, g_w_z[:, 2:3] * s2

        inv_re = 1.0 / self.re_number
        residuals: Dict[str, torch.Tensor] = {
            "continuity": u_x + v_y + w_z,
            "momentum_x": u * u_x + v * u_y + w * u_z + p_x - inv_re * (u_xx + u_yy + u_zz),
            "momentum_y": u * v_x + v * v_y + w * v_z + p_y - inv_re * (v_xx + v_yy + v_zz),
            "momentum_z": u * w_x + v * w_y + w * w_z + p_z - inv_re * (w_xx + w_yy + w_zz),
        }

        if self.output_dim == 5:
            if self.pe_number is None or self.pe_number <= 0:
                raise ValueError("A positive Peclet number is required when concentration c is present.")
            c = q[:, 4:5]
            gc = self._grad(c, x)
            c_x, c_y, c_z = gc[:, 0:1] * s1, gc[:, 1:2] * s1, gc[:, 2:3] * s1
            g_c_x = self._grad(gc[:, 0:1], x)
            g_c_y = self._grad(gc[:, 1:2], x)
            g_c_z = self._grad(gc[:, 2:3], x)
            c_xx = g_c_x[:, 0:1] * s2
            c_yy = g_c_y[:, 1:2] * s2
            c_zz = g_c_z[:, 2:3] * s2
            residuals["convection_diffusion"] = (
                u * c_x + v * c_y + w * c_z - (1.0 / self.pe_number) * (c_xx + c_yy + c_zz)
            )

        return q, residuals


class OperatorGeometryTransfer:
    def __init__(
        self,
        device: torch.device,
        standardizer: FieldStandardizer,
        lambda_data: float = 50.0,
        lambda_pde: float = 1.0,
        collocation_points: int = 20_000,
        seed: int = 42,
    ):
        self.device = device
        self.standardizer = standardizer
        self.lambda_data = float(lambda_data)
        self.lambda_pde = float(lambda_pde)
        self.collocation_points = int(collocation_points)
        self.rng = np.random.default_rng(seed)

    def hot_start_target_model(
        self,
        target_model: GD_PINNs_Net,
        source_state_1nn: Dict[str, torch.Tensor],
        neighbor_states: Sequence[Dict[str, torch.Tensor]],
        weights: np.ndarray,
    ) -> GD_PINNs_Net:
        """
        Exact 1-NN inheritance for the operator encoder. FiLM is inherited from the same
        aligned source because it acts on the inherited latent channels. Only the geometry
        adapter is graph-weighted, matching the manuscript's explicit interpolation rule.
        """
        target_state = target_model.state_dict()
        for key in list(target_state.keys()):
            if key in ("field_mean", "field_std"):
                continue
            if key.startswith("encoder.") or key.startswith("film_mlp."):
                target_state[key] = source_state_1nn[key].detach().clone()
            elif key.startswith("adapter."):
                fused = torch.zeros_like(target_state[key])
                for w, state in zip(weights, neighbor_states):
                    fused = fused + float(w) * state[key].to(fused.device)
                target_state[key] = fused
        target_model.load_state_dict(target_state, strict=True)
        target_model.set_field_standardizer(self.standardizer)
        return target_model

    def _sample_collocation(self, coords_ref: np.ndarray) -> np.ndarray:
        n = len(coords_ref)
        if n <= self.collocation_points:
            return coords_ref
        idx = self.rng.choice(n, size=self.collocation_points, replace=False)
        return coords_ref[idx]

    def _merge_boundary_constraints(
        self,
        data_coords: np.ndarray,
        data_fields: np.ndarray,
        raw_coords_ref: np.ndarray,
        raw_fields: np.ndarray,
        boundary_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if boundary_mask is None or not np.any(boundary_mask):
            return data_coords, data_fields
        return (
            np.concatenate([data_coords, raw_coords_ref[boundary_mask]], axis=0),
            np.concatenate([data_fields, raw_fields[boundary_mask]], axis=0),
        )

    def evaluate_model(
        self,
        model: GD_PINNs_Net,
        coords_ref: np.ndarray,
        raw_fields: np.ndarray,
        h_graph: torch.Tensor,
        max_eval_points: int = 200_000,
    ) -> float:
        if len(coords_ref) > max_eval_points:
            idx = self.rng.choice(len(coords_ref), size=max_eval_points, replace=False)
            coords_ref = coords_ref[idx]
            raw_fields = raw_fields[idx]
        X = torch.tensor(coords_ref, dtype=torch.float32, device=self.device)
        Y = torch.tensor(raw_fields, dtype=torch.float32, device=self.device)
        H = h_graph.unsqueeze(0).expand(len(X), -1)
        model.eval()
        with torch.no_grad():
            pred = model.forward_physical(X, H)
            err = torch.linalg.vector_norm(pred - Y, dim=0)
            denom = torch.linalg.vector_norm(Y, dim=0).clamp_min(1.0e-8)
            rel = err / denom
        return float(rel.mean().item())

    def train_pinn(
        self,
        model: GD_PINNs_Net,
        raw_coords_ref: np.ndarray,
        raw_fields: np.ndarray,
        boundary_mask: Optional[np.ndarray],
        data_coords_ref: np.ndarray,
        data_fields: np.ndarray,
        l_ref: float,
        h_graph: torch.Tensor,
        adam_epochs: int,
        lbfgs_iters: int,
        learning_rate: float,
    ) -> GD_PINNs_Net:
        data_coords_ref, data_fields = self._merge_boundary_constraints(
            data_coords_ref, data_fields, raw_coords_ref, raw_fields, boundary_mask
        )
        data_fields_std = self.standardizer.transform(data_fields)
        colloc = self._sample_collocation(raw_coords_ref)

        X_data = torch.tensor(data_coords_ref, dtype=torch.float32, device=self.device)
        Y_data = torch.tensor(data_fields_std, dtype=torch.float32, device=self.device)
        X_phys = torch.tensor(colloc, dtype=torch.float32, device=self.device)

        H_data = h_graph.unsqueeze(0).expand(len(X_data), -1)
        H_phys = h_graph.unsqueeze(0).expand(len(X_phys), -1)

        model.set_field_standardizer(self.standardizer)
        model.train()
        adam = optim.Adam(model.parameters(), lr=learning_rate)

        def compute_loss() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pred_std = model(X_data, H_data)
            loss_data = torch.mean((pred_std - Y_data) ** 2)
            _, residuals = model.physics_residuals(X_phys, H_phys, l_ref)
            loss_pde = torch.stack([torch.mean(r ** 2) for r in residuals.values()]).sum()
            total = self.lambda_data * loss_data + self.lambda_pde * loss_pde
            return total, loss_data, loss_pde

        for epoch in range(max(0, adam_epochs)):
            adam.zero_grad(set_to_none=True)
            loss, l_data, l_pde = compute_loss()
            loss.backward()
            adam.step()
            if adam_epochs > 0 and (epoch + 1) % max(1, adam_epochs // 10) == 0:
                print(
                    f"      Adam {epoch + 1:5d}/{adam_epochs}: "
                    f"L={loss.item():.4e}, data={l_data.item():.4e}, pde={l_pde.item():.4e}"
                )

        if lbfgs_iters > 0:
            lbfgs = LBFGS(
                model.parameters(),
                max_iter=lbfgs_iters,
                tolerance_grad=1.0e-5,
                tolerance_change=1.0e-12,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                loss, _, _ = compute_loss()
                loss.backward()
                return loss

            lbfgs.step(closure)

        return model


# =============================================================================
# Cache utilities and descriptor preparation
# =============================================================================


def cache_path(cache_dir: Path, task_id: int, tag: str, resolution: int) -> Path:
    return cache_dir / f"case_{task_id}_{tag}_r{resolution}.npz"


def build_or_load_cartesian(
    task: TaskRecord,
    field_cols: Sequence[str],
    cache_dir: Path,
    tag: str,
    resolution: int,
    interp_neighbors: int,
    idw_power: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    path = cache_path(cache_dir, task.task_id, tag, resolution)
    if path.exists():
        d = np.load(path)
        return d["grid"], d["fields"], float(d["l_ref"])

    coords, fields, _ = load_task_csv(task, field_cols)
    coords_ref, l_ref, _ = nondimensionalize_coordinates(coords)
    grid, tensor = reconstruct_cartesian_idw(
        coords_ref,
        fields,
        resolution=resolution,
        k_neighbors=interp_neighbors,
        power=idw_power,
    )
    np.savez_compressed(path, grid=grid, fields=tensor, l_ref=np.asarray(l_ref))
    return grid, tensor, l_ref


def descriptor_from_cached_grid(
    grid_fields: np.ndarray,
    extractor: SpectralPhysicalExtractor,
    standardizer: Optional[FieldStandardizer] = None,
) -> np.ndarray:
    fields = grid_fields
    if standardizer is not None:
        fields = standardizer.transform(fields)
    return extractor.compute_features(fields)


def temperature_schedule(progress: float, tau_start: float = 2.0, tau_end: float = 0.1) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(tau_start * (tau_end / tau_start) ** progress)


def load_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# =============================================================================
# Main GD-PINNs pipeline
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="GD-PINNs: repaired manifold-guided transfer reference code")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--coarse_ratio", type=float, default=0.30)
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--centrality_alpha", type=float, default=0.65)
    parser.add_argument("--tau_anchor", type=float, default=0.01)
    parser.add_argument("--anchor_batch", type=int, default=10)
    parser.add_argument("--graph_epochs", type=int, default=200)
    parser.add_argument("--graph_lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda_data", type=float, default=50.0)
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    parser.add_argument("--collocation_points", type=int, default=20_000)
    parser.add_argument("--adam_anchor_epochs", type=int, default=10_000)
    parser.add_argument("--adam_target_epochs", type=int, default=10_000)
    parser.add_argument("--lbfgs_iters", type=int, default=2_000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--re", type=float, default=229.1451)
    parser.add_argument("--pe", type=float, default=None)
    parser.add_argument("--interp_neighbors", type=int, default=8)
    parser.add_argument("--idw_power", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(args.data_dir)
    field_cols = infer_field_columns(tasks[0].csv_path)
    for task in tasks[1:]:
        if infer_field_columns(task.csv_path) != field_cols:
            raise ValueError("All tasks must expose the same physical field columns.")
    output_dim = len(field_cols)
    if output_dim == 5 and (args.pe is None or args.pe <= 0):
        raise ValueError("CSV files contain concentration c; provide a positive --pe value.")

    coarse_res = coarse_resolution_from_ratio(args.resolution, args.coarse_ratio)
    effective_ratio = (coarse_res / args.resolution) ** 3
    print(f"Device: {device}")
    print(f"Tasks: {len(tasks)} | fields: {field_cols}")
    print(
        f"Cartesian resolution: fine={args.resolution}^3, coarse={coarse_res}^3 "
        f"(effective retention={effective_ratio:.3f})"
    )

    extractor = SpectralPhysicalExtractor()
    graph = GraphManifoldConstructor(
        device=device,
        k_neighbors=args.k_neighbors,
        output_dim=64,
        centrality_alpha=args.centrality_alpha,
    )

    # -------------------------------------------------------------------------
    # Stage 1A: every target is represented only by coarse Cartesian observations.
    # This avoids using a full target solution to localize that target on the manifold.
    # -------------------------------------------------------------------------
    print("\n[Stage 1A] Coarse spectral-physical descriptors")
    coarse_features = np.zeros((len(tasks), 150), dtype=np.float64)
    for task in tasks:
        _, coarse_tensor, _ = build_or_load_cartesian(
            task,
            field_cols,
            cache_dir,
            tag="coarse",
            resolution=coarse_res,
            interp_neighbors=args.interp_neighbors,
            idw_power=args.idw_power,
        )
        coarse_features[task.index] = extractor.compute_features(coarse_tensor)
        if (task.index + 1) % max(1, len(tasks) // 10) == 0 or task.index + 1 == len(tasks):
            print(f"  coarse descriptors: {task.index + 1}/{len(tasks)}")

    # -------------------------------------------------------------------------
    # Stage 1B/2A: preliminary coarse graph and centrality-guided nested anchors.
    # Fine descriptors are only computed for tasks after they become anchors.
    # -------------------------------------------------------------------------
    print("\n[Stage 2A] Centrality-guided expanding Anchor Support Set")
    ranking, _ = graph.preliminary_ranking(coarse_features)

    def get_fine_feature(idx: int) -> np.ndarray:
        task = tasks[idx]
        _, fine_tensor, _ = build_or_load_cartesian(
            task,
            field_cols,
            cache_dir,
            tag="fine",
            resolution=args.resolution,
            interp_neighbors=args.interp_neighbors,
            idw_power=args.idw_power,
        )
        return extractor.compute_features(fine_tensor)

    anchor_indices, _ = graph.expand_anchor_set(
        ranking,
        fine_feature_getter=get_fine_feature,
        tau_anchor=args.tau_anchor,
        batch_size=args.anchor_batch,
        min_batches=2,
    )
    anchor_set = set(map(int, anchor_indices.tolist()))
    print(f"  Selected anchors: {len(anchor_indices)}")

    # Anchor-support statistics are now fixed and shared by all experiments.
    standardizer = FieldStandardizer.fit_anchor_support(tasks, anchor_indices, field_cols)
    with open(out_dir / "field_standardizer.json", "w", encoding="utf-8") as f:
        json.dump(
            {"fields": field_cols, "mean": standardizer.mean.tolist(), "std": standardizer.std.tolist()},
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Stage 2B: final hybrid manifold. Anchors use fine features, targets coarse.
    # All fields are standardized using anchor-support statistics before extraction.
    # -------------------------------------------------------------------------
    print("\n[Stage 2B] Hybrid fine-anchor / coarse-target manifold + GAT")
    hybrid_features = np.zeros_like(coarse_features)
    for task in tasks:
        if task.index in anchor_set:
            _, tensor, _ = build_or_load_cartesian(
                task,
                field_cols,
                cache_dir,
                tag="fine",
                resolution=args.resolution,
                interp_neighbors=args.interp_neighbors,
                idw_power=args.idw_power,
            )
        else:
            _, tensor, _ = build_or_load_cartesian(
                task,
                field_cols,
                cache_dir,
                tag="coarse",
                resolution=coarse_res,
                interp_neighbors=args.interp_neighbors,
                idw_power=args.idw_power,
            )
        hybrid_features[task.index] = descriptor_from_cached_grid(tensor, extractor, standardizer)

    embeddings = graph.build_and_train_graph(hybrid_features, epochs=args.graph_epochs, lr=args.graph_lr)
    torch.save(embeddings.detach().cpu(), out_dir / "graph_embeddings.pt")
    np.save(out_dir / "hybrid_features.npy", hybrid_features)
    np.save(out_dir / "anchor_indices.npy", anchor_indices)

    transfer = OperatorGeometryTransfer(
        device=device,
        standardizer=standardizer,
        lambda_data=args.lambda_data,
        lambda_pde=args.lambda_pde,
        collocation_points=args.collocation_points,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # Stage 3: train high-fidelity Anchor Support Set.
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Fine-grid anchor training")
    for number, anchor_idx in enumerate(anchor_indices, start=1):
        task = tasks[int(anchor_idx)]
        print(f"  Anchor {number}/{len(anchor_indices)} | task {task.task_id}")
        raw_coords, raw_fields, boundary = load_task_csv(task, field_cols)
        raw_coords_ref, l_ref, _ = nondimensionalize_coordinates(raw_coords)
        fine_grid, fine_tensor, _ = build_or_load_cartesian(
            task,
            field_cols,
            cache_dir,
            tag="fine",
            resolution=args.resolution,
            interp_neighbors=args.interp_neighbors,
            idw_power=args.idw_power,
        )
        data_coords = fine_grid.reshape(-1, 3)
        data_fields = fine_tensor.reshape(-1, output_dim)

        model = GD_PINNs_Net(
            output_dim=output_dim,
            graph_emb_dim=embeddings.shape[1],
            re_number=args.re,
            pe_number=args.pe,
        ).to(device)
        model.set_field_standardizer(standardizer)
        h = embeddings[int(anchor_idx)].detach()
        model = transfer.train_pinn(
            model,
            raw_coords_ref,
            raw_fields,
            boundary,
            data_coords,
            data_fields,
            l_ref,
            h,
            adam_epochs=args.adam_anchor_epochs,
            lbfgs_iters=args.lbfgs_iters,
            learning_rate=args.learning_rate,
        )
        rel_l2 = transfer.evaluate_model(model, raw_coords_ref, raw_fields, h)
        print(f"      anchor relative L2 = {rel_l2:.4e}")
        torch.save(model.state_dict(), out_dir / f"task_{task.task_id}_anchor.pth")

    # -------------------------------------------------------------------------
    # Stage 4: graph-diffusion progressive operator-geometry transfer.
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Progressive manifold-conditioned transfer")
    trained_indices: List[int] = list(map(int, anchor_indices.tolist()))
    remaining_indices: List[int] = [i for i in range(len(tasks)) if i not in anchor_set]
    initial_remaining = max(1, len(remaining_indices))

    while remaining_indices:
        target_idx = graph.select_next_task(trained_indices, remaining_indices)
        task = tasks[target_idx]
        solved_fraction = 1.0 - len(remaining_indices) / initial_remaining
        tau = temperature_schedule(solved_fraction, 2.0, 0.1)

        nearest_idx, neighbor_idx, weights = graph.get_transfer_weights(
            target_idx,
            trained_indices,
            tau=tau,
        )
        print(
            f"  Target task {task.task_id} | source={tasks[nearest_idx].task_id} | "
            f"neighbors={[tasks[i].task_id for i in neighbor_idx]} | tau={tau:.3f}"
        )

        neighbor_states: List[Dict[str, torch.Tensor]] = []
        for ni in neighbor_idx:
            neighbor_task = tasks[ni]
            migrated = out_dir / f"task_{neighbor_task.task_id}_migrated.pth"
            anchor = out_dir / f"task_{neighbor_task.task_id}_anchor.pth"
            path = migrated if migrated.exists() else anchor
            if not path.exists():
                raise FileNotFoundError(f"No trained state found for solved task {neighbor_task.task_id}")
            neighbor_states.append(load_state(str(path), device))

        state_1nn = neighbor_states[neighbor_idx.index(nearest_idx)]
        model = GD_PINNs_Net(
            output_dim=output_dim,
            graph_emb_dim=embeddings.shape[1],
            re_number=args.re,
            pe_number=args.pe,
        ).to(device)
        model.set_field_standardizer(standardizer)
        model = transfer.hot_start_target_model(model, state_1nn, neighbor_states, weights)

        raw_coords, raw_fields, boundary = load_task_csv(task, field_cols)
        raw_coords_ref, l_ref, _ = nondimensionalize_coordinates(raw_coords)
        coarse_grid, coarse_tensor, _ = build_or_load_cartesian(
            task,
            field_cols,
            cache_dir,
            tag="coarse",
            resolution=coarse_res,
            interp_neighbors=args.interp_neighbors,
            idw_power=args.idw_power,
        )
        data_coords = coarse_grid.reshape(-1, 3)
        data_fields = coarse_tensor.reshape(-1, output_dim)
        h = embeddings[target_idx].detach()

        model = transfer.train_pinn(
            model,
            raw_coords_ref,
            raw_fields,
            boundary,
            data_coords,
            data_fields,
            l_ref,
            h,
            adam_epochs=args.adam_target_epochs,
            lbfgs_iters=args.lbfgs_iters,
            learning_rate=args.learning_rate,
        )
        rel_l2 = transfer.evaluate_model(model, raw_coords_ref, raw_fields, h)
        print(f"      target relative L2 = {rel_l2:.4e}")
        torch.save(model.state_dict(), out_dir / f"task_{task.task_id}_migrated.pth")

        trained_indices.append(target_idx)
        remaining_indices.remove(target_idx)

    config = vars(args).copy()
    config.update(
        {
            "field_cols": field_cols,
            "coarse_resolution": coarse_res,
            "effective_coarse_ratio": effective_ratio,
            "n_tasks": len(tasks),
            "n_anchors": len(anchor_indices),
            "device": str(device),
        }
    )
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("\nGD-PINNs pipeline completed.")


if __name__ == "__main__":
    main()
