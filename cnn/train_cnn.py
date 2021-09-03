

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.cnn1 = nn.Conv2d(3, 64, 4,4,1)
        self.batch1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2,2,0)
        self.cnn2 = nn.Conv2d(64, 128, 4,4,1 )
        self.batch2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2,2,0)
        self.cnn3 = nn.Conv2d(128, 64, 4,2,1 )
        self.batch3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2,2,0)


        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64*2*2, 64*2)
        self.fc2 = nn.Linear(64*2,64)
        self.fc3 = nn.Linear(64,32)
        self.out = nn.Linear(32,1)


    def forward(self,x):
        ### Feature extraction
        x = self.cnn1(x)
        x = F.relu(self.batch1(x))
        
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = F.relu(self.batch2(x))
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = F.relu(self.batch3(x))
        x = self.maxpool3(x)

        x = self.flatten(x)

        ## Linear regression
        x = F.dropout(F.relu(self.fc1(x)), 0.2)
        x = F.dropout(F.relu(self.fc2(x)),0.2)
        x = F.dropout(F.relu(self.fc3(x)),0.2)
        out = F.relu(self.out(x))



        return out


device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")

model = CNN().to(device)

print('Model design')
summary(model,(3,512,512))



print("Loding data....")


class GraphsDataset(Dataset):
    def __init__(self, array_dir, root_dir, transform = None):

        self.n_nodes = np.load(array_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.n_nodes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_name =os.path.join(self.root_dir, "Image_{}.jpg".format(idx))
        # image = io.imread(img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
    
        n = torch.tensor(self.n_nodes[idx])
        sample = {'image' : image, 'n_nodes' : n}
        return sample


data_location =  input("Enter File Location: ")


traindata = GraphsDataset( 'data/' +data_location+  '/Training/n_nodes_train.npy', 'data/' +data_location + '/Training'
                   ,transform=transforms.Compose([
                               transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

valdata = GraphsDataset('data/' +data_location + '/Val/n_nodes_val.npy','data/' +data_location + '/Val'
                   ,transform=transforms.Compose([
                               transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
                   
trainloader = torch.utils.data.DataLoader(traindata, shuffle=True, batch_size = 256)
print('Train Data Loaded')
valloader = torch.utils.data.DataLoader(valdata, shuffle=True, batch_size = 256)
print('Validation Data Loaded')


print("Loading Done")

model = CNN().to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.000001, momentum = 0.8)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.MSELoss()
scheduler = StepLR(optimizer, step_size = 15, gamma = 0.5)


running_loss =0.0
max_epoch = input('Number of Epochs :')
try:
    max_epoch = int(max_epoch)
except ValueError:
    print("That's not an integer! (Number of Epochs)")
train_loss = []
val_loss = []
for epoch in range(max_epoch):
    trainbatch_loss = []
    valbatch_loss = []
    for i, data in enumerate( trainloader,0):
        x_train = data['image']
        y_train = data['n_nodes']
        x_train = x_train.to(device)
        y_train = y_train.to(device)


        optimizer.zero_grad()
        pred = model(x_train)
        pred = pred.flatten()

        loss = criterion(pred,y_train.float())
        running_loss += loss.item()
        trainbatch_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        

    scheduler.step()
    
    model.eval()
    for i, val_batch in enumerate( valloader,0):
        x_val = val_batch['image']
        y_val = val_batch['n_nodes']

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        pred_val = model(x_val)
        pred_val = pred_val.flatten()
        vloss = criterion(pred_val,y_val)
        valbatch_loss.append(vloss.item())

    train_epoch_loss = np.mean(trainbatch_loss)
    val_epoch_loss = np.mean(valbatch_loss)


    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    print('\n ','Epoch' , epoch +1, 'Training Loss :' , train_epoch_loss ,'|' , 'Validation Loss :', val_epoch_loss)

if os.path.isdir('models/'+data_location) == False :#
    os.mkdir('models/'+data_location)
torch.save(model, 'models/'+data_location+'/model_save')

np.save('models/' + data_location + '/Train_loss.npy', np.array(train_loss))
np.save('models/' + data_location + '/Val_loss.npy',np.array(val_loss))
