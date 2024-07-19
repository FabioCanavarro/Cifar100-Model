import torch
import torchvision as vision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
datasettrain = vision.datasets.CIFAR100("data",True,vision.transforms.ToTensor(),download=False)

datasettest = vision.datasets.CIFAR100("data",False,vision.transforms.ToTensor(),download=False)
datagen = torch.utils.data.DataLoader(datasettrain,64,True)
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
            nn.Conv2d(3,32,2),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            nn.Conv2d(32,64,2),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            nn.Conv2d(64,128,2),
            nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            nn.Conv2d(128,256,2),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            nn.Conv2d(256,512,2),
            nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            
            nn.Conv2d(512,1024,2),
            nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            
            nn.Conv2d(1024,2048,2),
            nn.BatchNorm2d(2048),
            torch.nn.ReLU(),
            
            nn.Conv2d(2048,4096,2),
            nn.BatchNorm2d(4096),
            torch.nn.ReLU(),
            
            nn.Conv2d(4096,(4096*2),2),
            nn.BatchNorm2d((4096*2)),
            torch.nn.ReLU(),
            
            nn.Conv2d((4096*2),(4096*4),2),
            nn.BatchNorm2d((4096*4)),
            torch.nn.ReLU(),
            
            nn.Conv2d((4096*4),(4096*8),2),
            nn.BatchNorm2d((4096*8)),
            torch.nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linearstack = nn.Sequential(
            nn.Linear((4096*8),100),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv2d(x)
        flat = self.flatten(x)
        logits = self.linearstack(flat)
        return logits
model = NN().to(device)
print(model)