import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# 1. فئة تحميل البيانات (NPZ Dataset Loader)
# ---------------------------------------------------------------------------
class SignLanguageNPZDataset(Dataset):
    """Dataset class for loading sign language data from NumPy NPZ archive."""
    
    def __init__(self, npz_file: str) -> None:
        """
        Initialize the dataset.
        
        Args:
            npz_file: Path to the NPZ file containing 'data' and 'labels'.
        """
        data = np.load(npz_file, allow_pickle=True)
        self.features = data['data'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


# ---------------------------------------------------------------------------
# 2. طبقة الرسم البياني العصبي المخصصة (Custom GCN Layer in Pure PyTorch)
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """
    Symmetric normalized Graph Convolutional layer in pure PyTorch.
    Formula: H^(l+1) = activation( D^(-1/2) * A_tilde * D^(-1/2) * H^l * W^l )
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input node features of shape (batch_size, num_nodes, in_features)
            adj_norm: Normalized adjacency matrix of shape (num_nodes, num_nodes)
        """
        # 1. Projection: xW (shape: batch_size, num_nodes, out_features)
        x = self.linear(x)
        
        # 2. Message passing: D^(-1/2) * A_tilde * D^(-1/2) * (xW)
        # Using batch matrix multiplication (B, N, N) @ (B, N, F_out)
        out = torch.matmul(adj_norm, x) + self.bias
        return out


# ---------------------------------------------------------------------------
# 3. موديل معالجة شبكة اليد (Sign Language GCN Model)
# ---------------------------------------------------------------------------
class SignLanguageGCNModel(nn.Module):
    """
    Graph Convolutional Network for hand landmark ASL classification.
    Features dynamic scaling and translation normalization inside the forward pass.
    """
    def __init__(self, in_features: int = 3, num_nodes: int = 21, 
                 hidden_dim: int = 128, num_classes: int = 24):
        super(SignLanguageGCNModel, self).__init__()
        self.num_nodes = num_nodes
        
        # Register static normalized adjacency matrix as a buffer (saved with state_dict but not learnable)
        self.register_buffer('adj_norm', self._get_normalized_adj())
        
        # GCN blocks
        self.gc1 = GraphConv(in_features, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim * 2)
        self.gc3 = GraphConv(hidden_dim * 2, hidden_dim)
        
        # Normalization and activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.bn3 = nn.BatchNorm1d(num_nodes)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _get_normalized_adj(self) -> torch.Tensor:
        """Computes the symmetric normalized adjacency matrix D^(-1/2) * A_tilde * D^(-1/2) for a hand skeleton."""
        # 21 nodes (0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky)
        connections = [
            # Wrist to finger roots
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # Thumb
            (1, 2), (2, 3), (3, 4),
            # Index
            (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20)
        ]
        
        # Create symmetric adjacency matrix A
        A = np.zeros((21, 21), dtype=np.float32)
        for i, j in connections:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
        # Add self-loops (A_tilde = A + I)
        A_tilde = A + np.eye(21, dtype=np.float32)
        
        # Compute D_tilde^(-1/2)
        row_sum = A_tilde.sum(axis=1)
        d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum>0)
        d_inv_sqrt[row_sum <= 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
        adj_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        return torch.tensor(adj_norm, dtype=torch.float32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 63)
        # Reshape to (batch_size, 21 nodes, 3 features [x, y, z])
        x = x.view(-1, self.num_nodes, 3)
        
        # ─── Dynamic Center and Scale Normalization (Differentiable & Robust) ───
        # 1. Translation invariance: Center around wrist (node 0)
        wrist = x[:, 0:1, :]  # shape: (B, 1, 3)
        x = x - wrist
        
        # 2. Scale invariance: Divide by distance between Wrist (0) and Middle finger MCP (9)
        middle_mcp = x[:, 9, :]  # shape: (B, 3)
        scale = torch.norm(middle_mcp, dim=1, keepdim=True).unsqueeze(-1)  # shape: (B, 1, 1)
        scale = torch.clamp(scale, min=1e-6)  # Avoid division by zero
        x = x / scale
        
        # ─── Graph Convolutional Blocks ───
        # Block 1
        h = self.gc1(x, self.adj_norm)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        # Block 2
        h = self.gc2(h, self.adj_norm)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        # Block 3
        h = self.gc3(h, self.adj_norm)
        h = self.bn3(h)
        h = self.relu(h)
        
        # Flatten features (shape: batch_size, hidden_dim * num_nodes)
        h = h.view(h.size(0), -1)
        
        # Classification head
        out = self.classifier(h)
        return out


# ---------------------------------------------------------------------------
# 4. دالة تدريب الموديل (GCN Training Pipeline)
# ---------------------------------------------------------------------------
def train_gcn_model(npz_file: str, epochs: int = 100, batch_size: int = 64, 
                    learning_rate: float = 0.001) -> None:
    """Trains the hand landmark GCN recognition model."""
    import matplotlib.pyplot as plt
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("Loading dataset...")
    dataset = SignLanguageNPZDataset(npz_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the GCN Model
    model = SignLanguageGCNModel(num_classes=24).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping config
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss /= total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping & model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best weights
            torch.save(model.state_dict(), "sign_language_gcn_model.pth")
            print("--> Checkpoint saved: 'sign_language_gcn_model.pth'")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # ---------------------------------------------------------------------------
    # 5. رسم منحنيات التعلم وحفظها كصورة (Plot Learning Curves)
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='#1f77b4', lw=2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='#ff7f0e', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='#1f77b4', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='#ff7f0e', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('gcn_training_curves.png')
    plt.close()
    print("Training curves saved as 'gcn_training_curves.png'")
    print("GCN Training complete.")


if __name__ == "__main__":
    import os
    
    # List of candidate paths to search for the dataset (Local & Kaggle)
    candidate_paths = [
        "landmarks_dataset.npz",  # Local/CWD
        "/kaggle/input/datasets/aa1bee/landmarkdataset/landmarks_dataset.npz",  # User's Kaggle path
        "/kaggle/input/landmarkdataset/landmarks_dataset.npz",  # Alternative Kaggle path
    ]
    
    npz_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            npz_path = path
            break
            
    if npz_path:
        print(f"Dataset successfully found at: {npz_path}")
        train_gcn_model(npz_path, epochs=100, batch_size=64, learning_rate=0.001)
    else:
        print("Error: Dataset 'landmarks_dataset.npz' not found in any of the expected paths:")
        for path in candidate_paths:
            print(f"  - {path}")
        print("\nPlease verify you have uploaded the dataset and set the correct path in the script!")
