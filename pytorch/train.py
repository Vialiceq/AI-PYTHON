import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from conv2d_model import *
import time
from torch.utils.tensorboard import SummaryWriter


#DATA
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length
train_set_size = len(train_set)
test_set_size = len(test_set)

print("train_set length:{}".format(train_set_size))
print("test_set length:{}".format(test_set_size))

#load date
train_dataloader = DataLoader(train_set,64)
test_dataloader = DataLoader(test_set,64)


#bind device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creat model proto
neural1 = neural()
neural1.to(device)

#lose Func
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#optim
learning_rate = 1e-3
optimizer = torch.optim.SGD(neural1.parameters(), lr=learning_rate)


#record
total_train_steps = 0
total_test_steps = 0
epoch = 50


#add tensorboard
writter = SummaryWriter("./train-logs")

for i in range(epoch):
    print("-------------round {} training strat---------".format(i+1))
    start_time = time.time()
    #train start
    neural1.train()
    for data in train_dataloader:
        imgs, trargets = data
        imgs = imgs.to(device)
        targets = trargets.to(device)
        outputs = neural1(imgs).to(device)
        loss = loss_fn(outputs, targets)
        
        #optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_steps += 1
        if total_train_steps%100 == 0:
            print("training times: {},loss:{}".format(total_train_steps,loss.item()))
            writter.add_scalar("train_loss", loss.item(),total_train_steps)
        
    
    #test
    #neural1.eval()
    total_loss_test=0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, trargets = data
            imgs = imgs.to(device)
            targets = trargets.to(device)
            outputs = neural1(imgs).to(device)
            loss = loss_fn(outputs, targets)
            total_loss_test += loss.item()
            acurracy = ((outputs.argmax(1)==targets).sum())
            total_accuracy += acurracy
            
    print("Total accuracy:{}".format(total_accuracy/test_set_size))
    print("total test loss: {}".format(total_loss_test)) 
    writter.add_scalar("test loss",total_loss_test,total_test_steps )
    writter.add_scalar("test accuracy",total_accuracy/test_set_size,total_test_steps)
    total_test_steps += 1
    #save 
    if i%10 == 0 :
        torch.save(neural1, "./model/nural_{}".format(i))
        #conut time       
        end_time = time.time()
        print(end_time - start_time)


writter.close()

  
