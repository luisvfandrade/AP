#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    
    def __init__(self, dropout_prob):
        super(CNN, self).__init__()
        # Batch size = batchSize, images 28x28 => x.shape = [batchSize, 1, 28, 28]
        
        #Calculate output size of convolution given same input height and width:
        #source: https://iq.opengenus.org/output-size-of-convolution/
        
        #outputHeight = (inputHeight + paddingHeightTop + paddingHeightBot - kernelHeight) / (strideHeight) + 1
        
        #the following assumptions are made since input's height = input's width
        #paddingHeightTop = paddingHeightBot = padding and
        #kernelHeight = kernelWidth = kernel and
        #strideHeight = strideWidth = stride hence,
        
        #outputHeight = (inputHeight + 2 x padding - kernel) / (stride) + 1
        
        #for inputHeight = 28, kernel = 5, stride = 1 and expected outputHeight = 28, padding should be 2
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=5, padding = 2, stride = 1)
        
        #x.shape = [batchSize, 8, 28, 28]
        
        self.maxpool1 = nn.MaxPool2d(2, stride = 2)
        
        #outputHeight = (28 - 2)/2 + 1 = 14 => x.shape = [batchSize, 8, 14, 14]
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, padding = 0, stride = 1)
        
        #outputHeight = 14 - 3 + 1 = 12 => x.shape = [batchSize, 16, 12, 12]
        
        self.maxpool2 = nn.MaxPool2d(2, stride = 2)
        
        #outputHeight = (12 - 2)/2 + 1 = 6 => x.shape = [batchSize, 16, 6, 6] 
        
        #number of input features = number of ouput channels x output width x output height = 16 * 6 * 6 = 576
        
        self.fc1 = nn.Linear(576, 600)
        
        self.conv_drop = nn.Dropout(p = dropout_prob)
        
        self.fc2 = nn.Linear(600, 120)
        
        self.fc3 = nn.Linear(120, 10)
        
        
    def forward(self, x):
        
        x = x.view(-1, 1, 28, 28)
                
        #The following x transformations are according to the given order of the assignment guidelines of exercise 2.4.:
        #"A rectified linear unit activation function."
        x = F.relu(self.conv1(x))
        
        #"A max pooling with kernel size 2x2 and stride of 2."
        x = self.maxpool1(x)
               
        #"A rectified linear unit activation function."
        x = F.relu(self.conv2(x))
        
        #"A max pooling with kernel size 2x2 and stride of 2."
        x = self.maxpool2(x)

        # Reshape => x.shape = [batchSize, 576]
        x = x.view(-1, 576)
        
        #"An affine transformation with 600 output features (...) A rectified linear unit activation function."
        x = self.fc1(x)
        x = F.relu(x)
        
        #"A dropout layer wit a dropout probability of 0.3."
        x = self.conv_drop(x)
        
        #"An affine transformation with 120 output features (...) A rectified linear unit activation function."
        x = self.fc2(x)
        x = F.relu(x)
        
        #"An affine transformation with the number of classes followed by an output LogSoftmax layer."
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        #missing view transofrmation for output?
        
        return x


def train_batch(X, y, model, optimizer, criterion, **kwargs):
        
    model.train()

    inputs = X
    labels = y
    
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)

    # always in training phase, perform backward prop and optimize
    loss.backward()
    optimizer.step()

    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_feature_maps(model, train_dataset):
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    data, _ = train_dataset[4]
    data.unsqueeze_(0)
    output = model(data)

    plt.imshow(data.reshape(28,-1)) 
    plt.savefig('original_image.pdf')

    k=0
    act = activation['conv1'].squeeze()
    fig,ax = plt.subplots(2,4,figsize=(12, 8))
    
    for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
            ax[i,j].imshow(act[k].detach().cpu().numpy())
            k+=1  
            plt.savefig('activation_maps.pdf') 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.8)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y
    
    # initialize the model
    model = CNN(opt.dropout)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    plot_feature_maps(model, dataset)

if __name__ == '__main__':
    main()
