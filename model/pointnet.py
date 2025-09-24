import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from configs.config import POINTNET_OUTPUT_DIM


class TNet2D(nn.Module):
    def __init__(self, k=2):  # For 2D inputs
        super(TNet2D, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(k, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, k * k)
        )

    def forward(self, x):
        # x: [B, N, k]
        B = x.size(0)
        x = self.mlp(x)           # [B, N, 1024]
        x = torch.max(x, dim=1)[0]  # [B, 1024]
        x = self.fc(x)            # [B, k*k]
        
        id_matrix = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + id_matrix  # Add identity to stabilize
        x = x.view(-1, self.k, self.k)  # [B, k, k]
        return x

class PointNet2DWithTransform(nn.Module):
    def __init__(self, input_dim=2, output_dim=POINTNET_OUTPUT_DIM):
        super().__init__()
        self.input_transform = TNet2D(k=input_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.input_transform2 = TNet2D(k=128)
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: [B, N, 2]
        trans = self.input_transform(x)     # [B, 2, 2]
        x = torch.bmm(x, trans)             # Apply transform: [B, N, 2]

        x = self.mlp1(x)                    # [B, N, 128]
        trans2 = self.input_transform2(x)   # [B, 128, 128]
        x = torch.bmm(x, trans2)            # Apply second transform: [B, N, 128]
        x = self.mlp2(x)                    # [B, N, 1024]
        x = torch.max(x, dim=1)[0]          # [B, 1024]
        x = self.fc(x)                      # [B, output_dim]
        # x = x.view(-1, self.num_lines, 3)               # [B, num_lines, 3]
        return x #x[:, :, :2], x[:, :, 2]  # return lines and confidence
    
def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Example usage
    model = PointNet2DWithTransform(input_dim=2)
    # get dataset path from arglist and load file into a dataloader and train the model
    dataset = np.load("dataset.npz", allow_pickle=True)
    points_raw = dataset['points']
    lines_raw = dataset['lines']
    data = []
    for p, l in zip(points_raw, lines_raw):
        p = np.array(p, dtype=np.float32)  # Ensure proper dtype
        l = np.array(l, dtype=np.float32)
        data.append((torch.from_numpy(p), torch.from_numpy(l)))

    # Split and wrap in DataLoaders
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    training_dataset, validation_dataset = random_split(data, [train_size, val_size])

    train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    train_model(model, train_loader)
    #test model
    evaluate_model(model, val_loader)
    # Save the trained model
    torch.save(model.state_dict(), "pointnet2d_transformer.pth")
    print("Model trained and saved as pointnet2d_transformer.pth")
