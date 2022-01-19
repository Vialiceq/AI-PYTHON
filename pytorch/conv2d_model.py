import torch
from torch import nn
#from torch.nn import Conv2d, MaxPool2d , Flatten , Linear, Sequential


#defin neural
class neural(nn.Module):
    def __init__(self):
        super(neural,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
   
    input = torch.ones((64,3,32,32))
    input.to(torch.float64)
    device = torch.device('cuda:0')
    input = input.cuda()
    neural1 = neural().to(device)
    
    output = neural1(input)
    print(output.device)
