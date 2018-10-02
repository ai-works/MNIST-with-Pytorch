import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.autograd import Variable

kwargs = {} # nur bei CUDA kwargs = {'num_workers': 1, 'pin_memory': True}
### Stellt Daten bereit in den Ordner Data 
train_data = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))])),
        batch_size=64, shuffle=True, **kwargs)

### Testdaten generieren
test_data = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))])),
        batch_size=64, shuffle=True, **kwargs)



class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fcl = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10) # 10 Outputneuronen
   
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.relu(x)
        print(x.size())
        exit()
    

model = Netz()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.8)


def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = nn.CrossEntropyLoss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoche: {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(train_data.dataset),
                    100 * batch_id / len(train_data), loss.data[0]))
        
for epoch in range(1, 30):
    train(epoch)
    

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

