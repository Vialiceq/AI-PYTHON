from PIL import Image
import torchvision
import torch 
import torch.nn as nn

img_path = "pytorch/imgs/air.JPG"

img = Image.open(img_path)
img = img.convert('RGB')
print(type(img)) 

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])

img = transform(img)



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

#load model
model = torch.load("pytorch\model\model_nural_50")

#reshape img 
img = torch.reshape(img, (1,3,32,32))

# transfer input to cuda to use cuda_model
device = torch.device('cuda')
img = img.to(device)

#do not forget this
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print("the image in class (number in tensor):",output.argmax())