import torch
import torch.nn as nn
import pickle

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) 
    
class LogisticRegression(nn.Module):
    def __init__(self, batch_size):
        super(LogisticRegression, self).__init__()
        self.name = 'LogisticRegression (L2 regularization)'
        self.batch_size = batch_size
        self.flatten = Flatten()
        self.fc1 = nn.Linear(64*25*25*3, 64*13)
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = out.reshape(self.batch_size,64,13)
        return out

class FullyConnected(nn.Module):
    def __init__(self, batch_size):
        super(FullyConnected, self).__init__()
        self.name = 'FullyConnected'
        self.batch_size = batch_size
        self.flatten = Flatten()
        self.fc1 = nn.Linear(64*25*25*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64*13)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.reshape(self.batch_size,64,13)
        return out
    
class CNN_NoDropout(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_NoDropout, self).__init__()
        self.name = 'CNN (Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64*19*19, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)

class CNN_BatchNorm(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_BatchNorm, self).__init__()
        self.name = 'CNN (BatchNorm, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64*19*19, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)
    
class CNN_Dropout(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_Dropout, self).__init__()
        self.name = 'CNN (Dropout, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p = 0.05))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(p = 0.05))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64*19*19, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.05))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)
    

class CNN_Dropout_BatchNorm(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_Dropout_BatchNorm, self).__init__()
        self.name = 'CNN (BatchNorm, Dropout, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p = 0.05))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p = 0.05))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64*19*19, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.05))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)
    
class CNN_BatchNormLessFiltersLastLayer(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_BatchNormLessFiltersLastLayer, self).__init__()
        self.name = 'CNN (BatchNorm, Less Filters Last Layer, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16*19*19, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)

class CNN_BatchNormLessFilters(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_BatchNormLessFilters, self).__init__()
        self.name = 'CNN (BatchNorm, Less Filters, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16*19*19, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)
    
class CNN_LessFilters(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN_LessFilters, self).__init__()
        self.name = 'CNN (Less Filters, Leaky ReLU)'
        self.batch_size=batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU())
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16*19*19, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 13))

    def forward(self, x):
        x = x.reshape(self.batch_size*64,3,25,25)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.reshape(self.batch_size,64,13)
        return(x)


# pickle the file
def save_model(model, name):
    pickle.dump(model, open(name, 'wb'))