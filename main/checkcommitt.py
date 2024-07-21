import torch
import torchvision as vision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
datasettrain = vision.datasets.CIFAR100("data",True,vision.transforms.ToTensor(),download=True)

datasettest = vision.datasets.CIFAR100("data",False,vision.transforms.ToTensor(),download=True)
datagen = torch.utils.data.DataLoader(datasettrain,10,True)
datatestgen = torch.utils.data.DataLoader(datasettest,10,True)
for X,y in datagen:
    print(X[0].shape)
    print(y.shape)
    break
# temp = torch.utils.data.DataLoader(datasettrain,999999,True)
# for x,y in temp:
#     print(np.unique(y).shape)
#     print(y.shape)
#     break

#(100,)
#torch.Size([50000])

# nn.functional.one_hot(y,100).shape
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(3,32,2,padding=1),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,2,padding=1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(64,128,2,padding=1),
            nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(128,256,2,padding=1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(256,512,2,padding=1),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(512,1024,2,padding=1),
            nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(1024,2048,2,padding=1),
            nn.BatchNorm2d(2048),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(2048,4096,2,padding=1),
            nn.BatchNorm2d(4096),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2),
            
            nn.Conv2d(4096,(4096*2),2,padding=1),
            nn.BatchNorm2d((4096*2)),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(2)
            
        )
        
        self.flatten = nn.Flatten()
        self.linearstack = nn.Sequential(
            nn.Linear(4096*2,100),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        x = self.conv2d(x)
        flat = self.flatten(x)
        logits = self.linearstack(flat)
        return logits
model = NN().to(device)
print(model)

for batch, (X, y) in enumerate(datagen):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        break
pred.shape

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(datagen, model, loss_fn, optimizer)
    test(datatestgen, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")