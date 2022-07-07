import numpy as np
import torch
import os
import pdb
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from RFF_Dataset import LoRa_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


def main():
    # Open the binary and extract the data into a Numpy array
    with open('IQ_1.dat', 'rb') as fid:
        data_array = np.fromfile(fid, complex)

    print(len(data_array))
    
    # Extract the real values from the complex array into another array
    reals_column = data_array.real

    length = len(reals_column)

    # Extract the imaginary values from the complex array into another array
    imaginary_column = data_array.imag

    i_c1 = imaginary_column[0:8195]

    print(i_c1)

    # Create new dictionaries to hold the imaginary lists and the real lists
    imaginary_dict = defaultdict(list)
    real_dict = defaultdict(list)

    
    # In a for loop create the window name
    # and add to the dictionary with the data_array range
    for i in range(0,3):
        i_string = f"i_c{i+1}"
        r_string = f"r_c{i+1}"
        if(i== 0):
                imaginary_dict[i_string].append(imaginary_column[i:(8190)])
                real_dict[r_string].append(reals_column[i:(8190)])
        else:
                imaginary_dict[i_string].append(imaginary_column[(i*8190):(i+1)*8190])
                real_dict[r_string].append(reals_column[(i*8190):(i+1)*8190])

    

    dct = {}

    name_dict = {'Filename':['device1_window1.npy','device1_window2.npy','device1_window3.npy','device1_window4.npy'],'ID': [1,2,3,4]}

    file_data = pd.DataFrame(name_dict)

    file_data.to_csv('/home/kalad/data/indoor/data_inside_day1__tran1_2.csv')
    
    # concatenate the two arrays as two columns in a matrix
    df1 = pd.DataFrame({'col1':real_dict['r_c1'],'col2':imaginary_dict['i_c1']})
    df2 = pd.DataFrame({'col1':real_dict['r_c2'],'col2':imaginary_dict['i_c2']})
    df3 = pd.DataFrame({'col1':real_dict['r_c3'],'col2':imaginary_dict['i_c3']})
    df4 = pd.DataFrame({'col1':real_dict['r_c4'],'col2':imaginary_dict['i_c4']})

    
    np.save(os.path.join('/home/kalad/data/indoor/','device1_window1'), df1)
    np.save(os.path.join('/home/kalad/data/indoor/','device1_window2'), df2)
    np.save(os.path.join('/home/kalad/data/indoor/','device1_window3'), df3)
    np.save(os.path.join('/home/kalad/data/indoor/','device1_window4'), df4)


    # Set the input batch size (number of rows of the matrix)
    batch_size=64

    loaded_dataset = LoRa_Dataset("/home/kalad/data/indoor/data_inside_day1__tran1_2.csv","/home/kalad/data/indoor")


    train_size = int(0.8 * len(loaded_dataset))
    test_size = len(loaded_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(loaded_dataset, [train_size, test_size])

    print(f"train size {train_size}")
    print(f"test size {test_size}")

    train_loader= torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
            num_workers = 12, pin_memory=True)

    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=12, pin_memory = True)


    #train_features, train_labels = next(iter(train_loader))


    #print(train_features.view(49170, 1))
    #train_features = train_features.view(1, 49170)
    #print(train_features[0])
    
    class RFFNet(nn.Module):
    
        def __init__(self):
            super(RFFNet, self).__init__()

            # (Input channels, output channels, (Kernel))
            self.conv1 = nn.Conv2d(1,16,(1,4))
            self.block1_norm = nn.BatchNorm2d(16)
            # ((Kernel size)(Stride))
            self.maxpool1 = MaxPool2d((1, 2),(1, 2))
            self.relu1 = nn.LeakyReLU()
 
            self.conv2 = nn.Conv2d(in_channels =16, out_channels = 24,
                    kernel_size =(1,4))
            self.block2_norm = nn.BatchNorm2d(24)
            self.maxpool2 = MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.relu2 = nn.LeakyReLU()

            self.conv3 = nn.Conv2d(in_channels =24, out_channels = 32
                    , kernel_size=(1,4))
            self.block3_norm = nn.BatchNorm2d(32)
            self.maxpool3 = MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.relu3 = nn.LeakyReLU()
 
 
            self.conv4 = nn.Conv2d(in_channels =32, out_channels = 48
                    , kernel_size=(1,4))
            self.block4_norm = nn.BatchNorm2d(48)
            self.maxpool4 = MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.relu4 = nn.LeakyReLU()
            
            self.conv5 = nn.Conv2d(in_channels =48, out_channels =64
                    , kernel_size=(1,4))
            self.block5_norm = nn.BatchNorm2d(64)
            self.maxpool5 = MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.relu5 = nn.LeakyReLU()

            #(channels in,channels out,(kernel))
            self.conv6 = nn.Conv2d(64,96,(1,4))
            self.block6_norm = nn.BatchNorm2d(96)
            self.flatten = nn.Flatten(1)
            self.maxpool6 = nn.AvgPool1d(10)
            
            self.fc1 = nn.Linear(4800,500)
            self.fc_relu = nn.LeakyReLU()
            self.dropout = nn.Dropout2d(0,50)
            self.fc2 = nn.Linear(in_features =500, out_features=25)
            self.softmax = nn.Softmax(dim =0)
             
        def forward(self,x):
      
            x = x[0:1,:,:]
            #print(f"before Conv Block 1 inputs = {x.shape}")
            x = self.conv1(x)
            print(f"before Norm 1 inputs = {x.shape}")
            x = self.block1_norm(x)
            print(f"before maxpool 1 inputs = {x.shape}")
            x = self.maxpool1(x)
            print(f"before relu1 inputs = {x.shape}")
            x = self.relu1(x)
            print(f"before conv2  inputs = {x.shape}")
            x = self.conv2(x)
            print(f"before Norm 2 inputs = {x.shape}")
            x = self.block2_norm(x)
            x = self.maxpool2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.block3_norm(x)
            x = self.maxpool3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.block4_norm(x)
            x = self.maxpool4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.block5_norm(x)
            x = self.maxpool5(x)
            x = self.relu5(x)
            x = self.conv6(x)
            x = self.block6_norm(x)
            x = self.flatten(x)
            x = self.maxpool6(x)
            x = self.fc1(x)
            x = self.fc_relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.softmax(x)
          
            return x

    device = torch.device('cpu')
    model = RFFNet().to(device).float()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    summary(model,input_size=(1,2,8195),batch_size=-1, device="cpu")
    lossFn = nn.NLLLoss()

    EPOCHS = 10
    network = RFFNet()
    network = network.float()

    # initialize a dictionary to store training history
    H = {
    	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
    }
    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    
    # loop over our epochs
    for e in range(0, EPOCHS):
        # set the model in training mode
        model.train()
        optimizer.zero_grad()
	# initialize the total training and validation loss
	# loop over the training set
        for n, (x, y) in enumerate (train_loader):
		    
            optimizer.zero_grad()
            output = network(x)
            # send the input to the device
	    # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(output, y)
	    # zero out the gradients, perform the backpropagation step,
	    # and update the weights

            loss.backward()
            optimizer.step()
	    # add the loss to the total training loss so far and
	    # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

if __name__ == "__main__":
    main()



