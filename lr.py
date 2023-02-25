import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data.iloc[:, :-1].values).float()
        self.y = torch.tensor(data.iloc[:, -1].values).long()
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

df = pd.read_csv('./embedding/train.txt', sep=" ", header=None)

tags = df[25].unique()
dict = {k: v for v, k in enumerate(tags)}
df[25] = df[25].map(pd.Series(dict))


train_data, test_data = train_test_split(df, test_size=0.2)

batch_size = 32

# Create dataloaders for train and test sets
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

# Define hyperparameters
input_dim = 25
output_dim = len(tags)
lr = 0.1
num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function and optimizer
model = LogisticRegression(input_dim, output_dim).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
    # Print average loss for epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss / len(train_dataloader)))

torch.save(model.state_dict(), './models/lr.pth')

# Test model
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))