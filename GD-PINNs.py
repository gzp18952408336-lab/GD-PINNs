import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
import glob
import copy
from torch.optim import LBFGS
from sklearn.preprocessing import MinMaxScaler
from scipy.fft import fft
from scipy.signal import welch

# PyG Imports
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data


# 1. 基础物理模型 (VPNSFnet)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(3, 120), nn.Tanh(),
            nn.Linear(120, 120), nn.Tanh(),
            nn.Linear(120, 120), nn.Tanh()
        )
        self.block2 = nn.Sequential(
            nn.Linear(120, 120), nn.Tanh(),
            nn.Linear(120, 120), nn.Tanh()
        )
        self.output = nn.Linear(120, 4)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        return self.output(h2)

class VPNSFnet(nn.Module):
    """
    Navier-Stokes Physics-Informed Neural Network
    Inputs: (x, y, z)
    Outputs: (u, v, w, p)
    """

    def __init__(self, Re=229.1451):
        super().__init__()
        self.model = MLP()
        self.Re = Re

    def forward(self, x):
        return self.model(x)

    def net_NS(self, x):
        return self.forward(x)

    def net_f_NS(self, x):
        """
        Calculates PDE residuals for Navier-Stokes equations.
        Ref: Eq. (1) in paper[cite: 47, 48].
        """
        x = x.clone().detach().requires_grad_(True)
        uvwp = self.forward(x)
        u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]

        def grads(y):
            return autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

        u_grads = grads(u)
        v_grads = grads(v)
        w_grads = grads(w)
        p_grads = grads(p)

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


# 2. 频谱特征提取 (Spectral Feature Extraction) - 保持不变
class SpectralFeatureExtractor:

    def __init__(self, n_freq_bins=20, n_pod_modes=10):
        self.n_freq_bins = n_freq_bins
        self.n_pod_modes = n_pod_modes

    def compute_spectral_features(self, data):
        input_features = []
        for i in range(3):
            signal = data[:, i]
            freqs, psd = welch(signal, nperseg=min(256, len(signal)))
            if len(psd) > self.n_freq_bins:
                psd = psd[:self.n_freq_bins]
            else:
                psd = np.pad(psd, (0, self.n_freq_bins - len(psd)))
            input_features.extend(psd)

        output_features = []
        for i in range(3, 7):
            signal = data[:, i]
            fft_vals = np.abs(fft(signal))
            fft_vals = fft_vals[:len(fft_vals) // 2]
            if len(fft_vals) > self.n_freq_bins:
                fft_vals = fft_vals[:self.n_freq_bins]
            else:
                fft_vals = np.pad(fft_vals, (0, self.n_freq_bins - len(fft_vals)))

            if np.sum(fft_vals) > 0:
                fft_vals = fft_vals / np.sum(fft_vals)
            output_features.extend(fft_vals)

        pod_features = self.compute_pod_spectrum(data[:, 3:7])

        expected_dim = self.n_freq_bins * 3 + self.n_freq_bins * 4 + self.n_pod_modes
        all_features = input_features + output_features + pod_features.tolist()

        # Dimension alignment
        if len(all_features) > expected_dim:
            all_features = all_features[:expected_dim]
        elif len(all_features) < expected_dim:
            all_features = all_features + [0.0] * (expected_dim - len(all_features))

        return np.array(all_features)

    def compute_pod_spectrum(self, output_data, K=10):
        try:
            U, S, Vt = np.linalg.svd(output_data, full_matrices=False)
            energy = S ** 2
            if energy.sum() > 0:
                energy = energy / energy.sum()
            else:
                energy = np.ones(len(S)) / len(S)
        except:
            energy = np.ones(K) / K

        if len(energy) < K:
            energy = np.pad(energy, (0, K - len(energy)))
        return energy[:K]


# 3. GNN 网络结构 (StructureAwareGNN)
class StructureAwareGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, heads=8):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=4, concat=True)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.sage = SAGEConv(hidden_dim, hidden_dim)

        predictor_input_dim = (hidden_dim * heads) + (hidden_dim * 4) + hidden_dim
        self.transfer_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.elu(self.gat1(x, edge_index, edge_weight))
        x2 = F.elu(self.gat2(x1, edge_index, edge_weight))
        x3 = F.elu(self.gat3(x2, edge_index, edge_weight))
        x_sage = F.elu(self.sage(x3, edge_index))

        x1_mean = x1.mean(dim=0, keepdim=True).expand(x3.size(0), -1)
        x2_mean = x2.mean(dim=0, keepdim=True).expand(x3.size(0), -1)
        x_combined = torch.cat([x1_mean, x2_mean, x_sage], dim=1)

        return self.transfer_predictor(x_combined)

    def predict_edge_weights(self, node_embeddings, edge_index):
        src, dst = edge_index
        src_emb, dst_emb = node_embeddings[src], node_embeddings[dst]
        edge_features = torch.cat([src_emb, dst_emb], dim=1)
        return self.edge_predictor(edge_features).squeeze()


# 4. 知识图谱核心逻辑 (AdvancedKnowledgeNetwork)
class AdvancedKnowledgeNetwork:
    """
    Manages graph construction, GNN training, and RWR-based transfer weighting.
    Fixes:
    1. Adds self-supervised GNN training (Learning Gap)[cite: 504, 505].
    2. Implements Random Walk with Restart for multi-hop weights (Logic Gap).
    """

    def __init__(self, input_dim, device, k_neighbors=30, hidden_dim=128, output_dim=64):
        self.device = device
        self.k_neighbors_graph_build = k_neighbors
        # GNN is now a trainable module
        self.gnn = StructureAwareGNN(input_dim, hidden_dim, output_dim).to(device)
        self.edge_index = None
        self.edge_weights = None
        self.task_embeddings = None
        self.dense_adj = None  # Added for RWR and Training
        self.raw_features = None  # Store for training loop

    def build_sparse_knowledge_graph(self, spectral_features):
        """Constructs the graph and prepares for GNN training."""
        print("Building Spectral-Physical Graph...")
        features_tensor = torch.tensor(np.array(spectral_features), dtype=torch.float32).to(self.device)
        self.raw_features = features_tensor  # Save for training

        # Cosine Similarity [cite: 458]
        norms = torch.norm(features_tensor, p=2, dim=1, keepdim=True)
        features_norm = features_tensor / (norms + 1e-8)
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        # k-NN Graph Construction
        num_nodes = len(features_tensor)
        values, indices = torch.topk(similarity_matrix, k=self.k_neighbors_graph_build + 1, dim=1)

        edge_src = []
        edge_dst = []
        edge_w = []

        for i in range(num_nodes):
            for j_idx, sim_val in zip(indices[i], values[i]):
                j = j_idx.item()
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_w.append(sim_val.item())

        self.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(self.device)
        self.edge_weights = torch.tensor(edge_w, dtype=torch.float32).to(self.device)

        # Prepare Dense Adjacency for RWR and Loss calculation
        self.dense_adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        self.dense_adj[self.edge_index[0], self.edge_index[1]] = self.edge_weights

        print(f"Graph constructed: {num_nodes} nodes. Ready for training.")

    def train_gnn_embeddings(self, epochs=200, lr=0.01):

        if self.edge_index is None: raise ValueError("Graph not built.")

        print(f"🏋️ Training GNN for {epochs} epochs to learn manifold structure...")
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr)
        self.gnn.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            z = self.gnn(self.raw_features, self.edge_index, self.edge_weights)

            # 2. Reconstruct Adjacency (Self-Supervised Loss)
            # We want dot(z_i, z_j) to approximate the spectral similarity A_ij
            z_norm = F.normalize(z, p=2, dim=1)
            adj_pred = torch.matmul(z_norm, z_norm.t())

            loss = F.mse_loss(adj_pred, self.dense_adj)

            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"   Epoch {epoch}/{epochs} | Reconstruction Loss: {loss.item():.6f}")

        self.gnn.eval()
        with torch.no_grad():
            self.task_embeddings = self.gnn(self.raw_features, self.edge_index, self.edge_weights)
        print("✅ GNN Training Complete. Embeddings updated.")
        return self.task_embeddings

    def _compute_rwr_scores(self, start_node_idx, restart_prob=0.15, steps=20):

        num_nodes = self.dense_adj.shape[0]

        row_sums = self.dense_adj.sum(dim=1, keepdim=True) + 1e-8
        P = self.dense_adj / row_sums

        # Initialize restart vector (target node)
        r = torch.zeros(num_nodes, device=self.device)
        r[start_node_idx] = 1.0

        # Restart vector e_i
        e = r.clone()

        # Iteration
        for _ in range(steps):
            # Note: In standard RWR, c is restart prob.
            # r = (1-c) * P' * r + c * e
            r = (1 - restart_prob) * torch.matmul(P.t(), r) + restart_prob * e

        return r

    def find_center_task(self):
        """Unchanged logic, but uses the TRAINED embeddings now."""
        if self.task_embeddings is None: raise ValueError("Embeddings not computed. Run train_gnn_embeddings first.")

        # Degree Centrality
        num_nodes = len(self.task_embeddings)
        degree = torch.zeros(num_nodes, device=self.device)
        src, _ = self.edge_index
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg_centrality = degree / (degree.max() + 1e-8)

        # Eigenvector Centrality
        eig_centrality = torch.ones(num_nodes, device=self.device)
        # Use dense adj for stability in power iteration
        adj = self.dense_adj if self.dense_adj is not None else torch.zeros((num_nodes, num_nodes), device=self.device)

        for _ in range(20):
            eig_centrality = torch.matmul(adj, eig_centrality)
            eig_centrality = eig_centrality / (torch.norm(eig_centrality) + 1e-8)

        # Unified Score: alpha * C_eig + (1-alpha) * C_deg [cite: 532]
        combined_score = 0.6 * eig_centrality + 0.4 * deg_centrality
        center_idx = torch.argmax(combined_score).item()

        print(f"🎯 Center Task Selected: {center_idx} (Score: {combined_score[center_idx]:.4f})")
        return center_idx

    def select_next_task(self, trained_indices, remaining_indices):
        if not remaining_indices: return None, 0.0

        num_nodes = len(self.task_embeddings)
        influence = torch.zeros(num_nodes, device=self.device)
        influence[trained_indices] = 1.0

        # Use dense matrix for influence propagation
        adj = self.dense_adj

        for _ in range(3):
            influence = torch.matmul(adj, influence)

        best_score = -1.0
        best_idx = None
        for idx in remaining_indices:
            score = influence[idx].item()
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def get_transfer_components(self, target_idx, trained_indices, k=8):

        if len(trained_indices) == 0: return [], []

        # 1. Identify Top-K Candidates from Trained Set based on TRAINED Embedding Similarity
        target_emb = self.task_embeddings[target_idx]
        candidates_emb = self.task_embeddings[trained_indices]

        # Cosine similarity in latent space
        sims = F.cosine_similarity(target_emb.unsqueeze(0), candidates_emb)

        # Select Top-K
        k_actual = min(k, len(trained_indices))
        top_k_vals, top_k_local_indices = torch.topk(sims, k_actual)
        neighbor_indices = [trained_indices[i] for i in top_k_local_indices.cpu().numpy()]

        # [FIX 2: Logic Gap] Compute RWR scores specifically for this target node
        # This gives the true multi-hop probability from target to all other nodes
        rwr_scores = self._compute_rwr_scores(target_idx)

        # 2. Compute Composite Weights (Eq. 16)
        weights = []
        target_neighbors = set(self.edge_index[1][self.edge_index[0] == target_idx].cpu().numpy())

        for i, src_idx in enumerate(neighbor_indices):
            # Term 1: Direct Similarity (Spectral) - From Embedding Sim
            w_direct = top_k_vals[i].item()

            # Term 2: Multi-hop (RWR Score)
            # Fetch the RWR probability of reaching src_idx starting from target_idx
            w_multihop = rwr_scores[src_idx].item()

            # Term 3: Structural Similarity (Jaccard) [cite: 559]
            src_neighbors = set(self.edge_index[1][self.edge_index[0] == src_idx].cpu().numpy())
            intersection = len(target_neighbors & src_neighbors)
            union = len(target_neighbors | src_neighbors)
            w_struct = intersection / union if union > 0 else 0.0

            # Composite Weight (Eq. 16 coefficients)
            # Alpha=0.5, Beta=0.3, Gamma=0.2
            w_final = 0.5 * w_direct + 0.3 * w_multihop + 0.2 * w_struct
            weights.append(max(0, w_final))

        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        return neighbor_indices, weights

# 5. 参数迁移 (Disk-based Transfer)
class MemoryEfficientTransfer:
    """
    Handles weighted parameter fusion by loading models from disk on-the-fly.
    Prevents VRAM explosion by never holding all models in memory.
    Ref: Eq. (15) Convex Combination[cite: 550].
    """

    def __init__(self, device):
        self.device = device

    def weighted_transfer(self, target_model, neighbor_indices, neighbor_weights, model_dir, task_map):
        print(f"   ⚖️ Fusing parameters from {len(neighbor_indices)} neighbors (Eq. 15)...")
        target_state = target_model.state_dict()

        # Initialize fused state with zeros
        fused_state = {k: torch.zeros_like(v).to(self.device) for k, v in target_state.items()}
        total_valid_weight = 0.0

        # No gradients needed during fusion
        with torch.no_grad():
            for idx, real_graph_idx in enumerate(neighbor_indices):
                weight = neighbor_weights[idx]
                if weight < 1e-4: continue

                real_task_id = task_map[real_graph_idx]

                # Construct possible file paths
                paths = [
                    os.path.join(model_dir, f"task_{real_task_id}_NS.pth"),
                    os.path.join(model_dir, f"task_{real_task_id}_migrated.pth")
                ]

                loaded = False
                for p in paths:
                    if os.path.exists(p):
                        # Load to CPU first to save GPU memory
                        src_state = torch.load(p, map_location=self.device)
                        for k in fused_state.keys():
                            if k in src_state:
                                fused_state[k] += weight * src_state[k]
                        total_valid_weight += weight
                        loaded = True
                        # Explicitly delete to free memory
                        del src_state
                        torch.cuda.empty_cache()
                        break

                if not loaded:
                    print(f"   ⚠️ Warning: Neighbor model {real_task_id} not found, skipping.")

        # Re-normalize if some models were missing
        if total_valid_weight > 0:
            for k in fused_state.keys():
                fused_state[k] /= total_valid_weight

        target_model.load_state_dict(fused_state)
        return target_model


# 6. 物理约束精细调整
class PhysicsTuner:
    def __init__(self, device):
        self.device = device

    def train(self, model, data, is_center=False):
        """
        Fine-tunes the model using strictly defined data ratios.
        Center task: 60% data. Transfer task: 20% data[cite: 569, 572].
        """
        ratio = 0.6 if is_center else 0.2
        epochs_adam = 5000 if is_center else 1500
        epochs_lbfgs = 50000 if is_center else 5000

        # Data Preparation
        total_len = len(data)
        if total_len == 0: return model

        n_sample = max(1, int(total_len * ratio))
        indices = np.random.choice(total_len, n_sample, replace=False)
        sampled_data = data[indices]

        # Split Physics (Collocation) and Boundary points (approx 80/20 split)
        split = int(n_sample * 0.8)
        X_phys = torch.tensor(sampled_data[:split, 0:3], dtype=torch.float32, requires_grad=True).to(self.device)
        X_b = torch.tensor(sampled_data[split:, 0:3], dtype=torch.float32).to(self.device)
        Y_b = torch.tensor(sampled_data[split:, 3:7], dtype=torch.float32).to(self.device)

        # 1. Adam Optimization
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        def compute_loss():
            # Boundary Loss
            pred_b = model.net_NS(X_b)
            loss_b = torch.mean((pred_b - Y_b) ** 2)

            # Physics Residual Loss (Eq. 19)
            _, f_u, f_v, f_w, f_e = model.net_f_NS(X_phys)
            loss_f = torch.mean(f_u ** 2 + f_v ** 2 + f_w ** 2 + f_e ** 2)

            return 100 * loss_b + loss_f

        # print(f"   Starting Adam ({epochs_adam} epochs)...")
        for _ in range(epochs_adam):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

        # 2. LBFGS Optimization
        # print(f"   Starting LBFGS...")
        lbfgs = LBFGS(model.parameters(), max_iter=epochs_lbfgs, line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            loss = compute_loss()
            loss.backward()
            return loss

        lbfgs.step(closure)
        return model


# 7. 主程序 (Main Execution)
if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 请根据实际情况修改以下路径
    DATA_ROOT = r"D:\task2\csv_cases"
    OUTPUT_DIR = r"D:\task2\output_models_optimized"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Components
    extractor = SpectralFeatureExtractor()
    # 150 dim = 60 (PSD) + 80 (FFT) + 10 (POD)
    knowledge_net = AdvancedKnowledgeNetwork(input_dim=150, device=DEVICE, k_neighbors=30)
    transfer_engine = MemoryEfficientTransfer(DEVICE)
    tuner = PhysicsTuner(DEVICE)

    # ---------------------------------------------------------
    # Step 1: Data Loading & Feature Extraction
    # ---------------------------------------------------------
    print(" Step 1: Extracting Features & Loading Data...")
    task_features = []
    task_data_map = {}  # Map internal graph index -> raw data
    task_id_map = {}  # Map internal graph index -> real task ID (from filename)

    # Scan directories
    case_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "case_*")))
    internal_idx = 0

    for case_dir in case_dirs:
        try:
            case_name = os.path.basename(case_dir)
            case_id = int(case_name.split('_')[1])
            csv_path = os.path.join(case_dir, f"{case_name}.csv")

            if not os.path.exists(csv_path): continue

            # Load Data
            data = np.loadtxt(csv_path, delimiter=',')
            if data.shape[0] < 50: continue  # Skip empty/small files

            # Extract Spectral Features
            feats = extractor.compute_spectral_features(data)

            task_features.append(feats)
            task_data_map[internal_idx] = data
            task_id_map[internal_idx] = case_id
            internal_idx += 1

            if internal_idx % 50 == 0: print(f"   Processed {internal_idx} tasks...")
        except Exception as e:
            print(f"   Error processing {case_dir}: {e}")

    num_tasks = len(task_features)
    print(f"✅ Total valid tasks: {num_tasks}")

    if num_tasks == 0:
        print(" No valid tasks found. Exiting.")
        exit()

    # ---------------------------------------------------------
    # Step 2: Build Graph & Select Center Task
    # ---------------------------------------------------------
    print("\n Step 2: Building Knowledge Graph...")
    knowledge_net.build_sparse_knowledge_graph(task_features)

    # [新增步骤] 训练 GNN 以学习流形结构
    # 这会更新 knowledge_net.task_embeddings
    knowledge_net.train_gnn_embeddings(epochs=200, lr=0.01)

    center_idx = knowledge_net.find_center_task()
    real_center_id = task_id_map[center_idx]
    print(f"   Center Task Real ID: {real_center_id}")

    # ---------------------------------------------------------
    # Step 3: Train Center Task
    # ---------------------------------------------------------
    print(f"\n Step 3: Training Center Task {real_center_id}...")
    center_model = VPNSFnet().to(DEVICE)

    # Check if pretrained exists
    pretrained_path = os.path.join(r"D:\task2\trained_models", f"task_{real_center_id}_NS.pth")
    if os.path.exists(pretrained_path):
        center_model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print("   Loaded pretrained weights.")

    center_model = tuner.train(center_model, task_data_map[center_idx], is_center=True)

    # SAVE to disk and DELETE from memory (Fixing VRAM issue)
    save_path = os.path.join(OUTPUT_DIR, f"task_{real_center_id}_NS.pth")
    torch.save(center_model.state_dict(), save_path)
    print(f"    Saved center model to {save_path}")

    del center_model
    torch.cuda.empty_cache()

    # Step 4: Progressive Transfer Loop

    print("\n Step 4: Starting Progressive Transfer...")

    trained_indices = [center_idx]
    remaining_indices = [i for i in range(num_tasks) if i != center_idx]

    loop_count = 0
    while remaining_indices:
        loop_count += 1
        print(f"\n--- Transfer Iteration {loop_count} ---")

        next_idx, score = knowledge_net.select_next_task(trained_indices, remaining_indices)
        if next_idx is None: break

        real_id = task_id_map[next_idx]
        print(f" Selected Task {real_id} (Influence Score: {score:.4f})")

        neighbor_indices, weights = knowledge_net.get_transfer_components(next_idx, trained_indices, k=8)
        print(f"   Neighbors (Graph IDs): {neighbor_indices}")
        print(f"   Weights: {np.round(weights, 3)}")

        new_model = VPNSFnet().to(DEVICE)
        new_model = transfer_engine.weighted_transfer(
            new_model, neighbor_indices, weights, OUTPUT_DIR, task_id_map
        )

        new_model = tuner.train(new_model, task_data_map[next_idx], is_center=False)

        save_path = os.path.join(OUTPUT_DIR, f"task_{real_id}_migrated.pth")
        torch.save(new_model.state_dict(), save_path)

        del new_model
        torch.cuda.empty_cache()

        trained_indices.append(next_idx)
        remaining_indices.remove(next_idx)

        print(f"✅ Task {real_id} completed. Progress: {len(trained_indices)}/{num_tasks}")

    print("\n All tasks completed successfully.")