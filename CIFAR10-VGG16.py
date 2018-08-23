import os
import torch
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models,datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time

# Check availability of GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    pinMem = True # Flag for pinning GPU memory
    print('GPU is available!')
else:
    pinMem = False

apply_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),])
                                       #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainLoader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./CIFAR10/', train=True, download=True,
                   transform = apply_transforms), batch_size=32, shuffle=True, num_workers=1, pin_memory=pinMem)
testLoader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./CIFAR10/', train=False,
                  transform = apply_transforms), batch_size=32, shuffle=True, num_workers=1, pin_memory=pinMem)


# Pre-trained VGG16
net = models.vgg16(pretrained=True)
net.classifier[6] = nn.Linear(4096,10) 
print(net)
if use_gpu:
    net = net.cuda()

criterion = nn.NLLLoss() # Negative Log-likelihood
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9) # Stochastic gradient descent

iterations = 2
trainLoss = []
testAcc = []
start = time.time()
for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0    
    net.train(True) # For training
    for i, data in enumerate(trainLoader, 0):
        inputs,labels = data
        # Wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)         
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Feed-forward input data through the network        
        outputs = net(inputs)
        # Compute loss/error
        loss = criterion(outputs, labels)
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.item()
        
    avgTrainLoss = runningLoss/50000.0
    trainLoss.append(avgTrainLoss)
    
    # Evaluating performance on test set for each epoch
    net.train(False) # For testing
    running_correct = 0
    for i, data in enumerate(testLoader, 0):

        inputs,labels = data
        # Wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            outputs = F.log_softmax(net(inputs),dim=1)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
        else:
            inputs = Variable(inputs)
            outputs = F.log_softmax(net(inputs),dim=1)
            _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum()
        
    avgTestAcc = running_correct.numpy()/10000.0
    testAcc.append(avgTestAcc)
   
    epochEnd = time.time()-epochStart
    print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f} ; Testing Acc: {:.3f} ; Time consumed: {:.0f}m {:.0f}s '\
          .format(epoch + 1,iterations,avgTrainLoss,avgTestAcc*100,epochEnd//60,epochEnd%60))
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))
