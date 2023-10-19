import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch_geometric.data import Data, DataLoader as GCNDataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)  # Add a linear layer to match the output dimensions

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.lin(x)  # Apply the linear layer to match the output dimensions
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the dataset
data_path = r"C:\Users\002YHJ744\Desktop\Personal\Reserach papers\p8\Code\Images\UCMerced_LandUse\UCMerced_LandUse\Images"  # Path to the UC Merced Land Use Dataset
dataset = ImageFolder(data_path, transform=transform)

# Create a GCN dataset
gcn_dataset = []
for i in range(len(dataset)):
    x, _ = dataset[i]
    num_nodes = x.size(0)
    edge_index = torch.zeros((2, num_nodes-1), dtype=torch.long)
    edge_index[0] = torch.arange(1, num_nodes)
    edge_index[1] = torch.zeros(num_nodes-1)
    y = torch.tensor([dataset.targets[i]], dtype=torch.long)
    gcn_dataset.append(Data(x=x.view(num_nodes, -1), edge_index=edge_index, y=y))  # Reshape x to (num_nodes, -1)

# Split the dataset into train and test sets
train_size = int(0.8 * len(gcn_dataset))
train_dataset = gcn_dataset[:train_size]
test_dataset = gcn_dataset[train_size:]

# Create GCN data loaders
train_loader = GCNDataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = GCNDataLoader(test_dataset, batch_size=1, shuffle=False)

# Create the GCN model
input_dim = dataset[0][0].view(-1).size(0)  # Flatten the input dimensions
hidden_dim = 64
output_dim = len(dataset.classes)
model = GCN(input_dim, hidden_dim, output_dim).to(device)

# Set training parameters
num_epochs = 10
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(output, data.y.squeeze().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")

# Evaluation
model.eval()
predictions = []
labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data.x.to(device), data.edge_index.to(device))
        preds = output.argmax(dim=1)
        predictions.append(preds.item())
        labels.append(data.y.item())

# Compute the false positive rate, true positive rate, and threshold for ROC curve
fpr, tpr, thresholds = roc_curve(labels, predictions)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
