import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# Network architecture.
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        hidden_1_size = 800
        hidden_2_size = 800
        self.fc1 = nn.Linear(28*28, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, 10)
        self.droput = nn.Dropout(0.5)
        
    def forward(self,input):
        output = input.view(-1, 28*28)
        output = F.relu(self.fc1(output))
        output = self.droput(output)
        output = F.relu(self.fc2(output))
        output = self.droput(output)
        output = self.fc3(output)
        return output


# Training algorithm.
def train(model, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_index, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        if batch_index % log_interval == 0:
            print(f'Training epoch {epoch}: '
                  + f'{batch_index * len(data)}/{len(train_loader.dataset)} '
                  + f'({100. * batch_index / len(train_loader):.0f}%)'
                  + f'\tLoss: {loss.item():.6f}')


# Testing algorithm.
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, labels, reduction='sum').item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set. \nAverage loss: {test_loss:.4f}. ' 
          + f'Accuracy: {correct}/{len(test_loader.dataset)} '
          + f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


# Initialize network.
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# Load dataset and prepare loaders.
batch_size = 20
transform = transforms.ToTensor() # convert data to torch.FloatTensor
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Train and test model.
epochs = 10
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
    scheduler.step()


# Print batches of images with predictions.
dataset_iter = iter(test_loader)
for batch in range(10):
    data, labels = next(dataset_iter)
    model.eval()
    correct = 0
    predicions = []
    with torch.no_grad():
        output = model(data)
        prediction = output.argmax(dim=1, keepdim=True)
        predicions.append(prediction)
        correct += prediction.eq(labels.view_as(prediction)).sum().item()
    print(f'Batch {batch}. Accuracy: {correct}/{batch_size} '
          + f'({100. * correct / batch_size:.0f}%).\n')
    image = plt.figure(figsize=(40, 10))
    for i in range(batch_size):
        handwritten_number = image.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
        handwritten_number.imshow(np.squeeze(data[i]), cmap='gray')
        handwritten_number.set_title(
            f'TRUE: {str(labels[i].item())} vs NET: {predicions[0][i][0]}')
    image.savefig(f'images/batch{str(batch)}')