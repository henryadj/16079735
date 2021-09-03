

import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import IPython.display as display
from tqdm import tqdm

### Code example taken from https://towardsdatascience.com/attention-in-computer-vision-fd289a5bd7ad
### Only the example fro the COnvolution Block Attention Module

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

query = torch.rand(128, 256, 24, 24)
attn = CBAM(gate_channels=256)
attn_output = attn(query)
print(attn_output.size())



class ConvPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1a = nn.Conv2d(3, 32, 5, padding=2)
        self.p1 = nn.MaxPool2d(2)
        self.c2a = nn.Conv2d(32, 32, 5, padding=2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 32, 5, padding=2)
        self.bn1a = nn.BatchNorm2d(32)
        self.bn2a = nn.BatchNorm2d(32)

    def forward(self, x):
        z = self.bn1a(F.leaky_relu(self.c1a(x)))
        z = self.p1(z)
        z = self.bn2a(F.leaky_relu(self.c2a(z)))
        z = self.p2(z)
        z = self.c3(z)
        return z

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvPart()
        self.final = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        z = self.conv(x)
        z = z.mean(3).mean(2)
        p = torch.sigmoid(self.final(z))[:, 0]
        return p, _

class NetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvPart()
        self.attn1 = CBAM(gate_channels=32)
        self.final = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        z = self.conv(x)
        q = self.attn1(z)
        z = q.mean(3).mean(2)
        p = torch.sigmoid(self.final(z))[:, 0]
        return p, q


## Custom data loader for the synthetic graph structure
class GraphsDataset(Dataset):
    def __init__(self, array_dir, root_dir, transform = None):

        self.labels = np.load(array_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        img_name =os.path.join(self.root_dir, "Image_{}.jpg".format(idx))
        # image = io.imread(img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        # image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        n = torch.tensor(self.labels[idx])

        sample = {'images' : image, 'labels' : n}
        return sample


save_location =  input("Enter File Location : ")

EPOCH = input('Number of Epochs :')
try:
    EPOCH = int(EPOCH)
except ValueError:
    print("That's not an integer! (Epoch)")

traindata = GraphsDataset('data/' + save_location + '/Train/Labels.npy', 'data/' + save_location +'/Train'
                   ,transform=transforms.Compose([
                               transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

valdata = GraphsDataset('data/' + save_location + '/Val/Labels.npy', 'data/' + save_location +'/Val'
                   ,transform=transforms.Compose([
                               transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

BATCH_SIZE = 128
trainloader = torch.utils.data.DataLoader(traindata, shuffle=True, batch_size = 128)
valloader = torch.utils.data.DataLoader(valdata, shuffle=True, batch_size = 128)




device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
print(device)

if os.path.isdir('models/' + save_location ) == False :
    os.mkdir('models/' + save_location )

if os.path.isdir('models/' + save_location + '/Results') == False :
    os.mkdir('models/' + save_location + '/Results')

def plot_without_attention(tr_err, ts_err, tr_acc, ts_acc, img):
    plt.clf()
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(tr_err, label='tr_err')
    axs[0].plot(ts_err, label='ts_err')
    axs[0].legend()
    axs[1].plot(tr_acc, label='tr_err')
    axs[1].plot(ts_acc, label='ts_err')
    axs[1].legend()
    axs[2].axis('off')
    axs[3].axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.01)

def plot_with_attention(tr_err, ts_err, tr_acc, ts_acc, img, att_out, no_images=6):
    plt.clf()
    fig, axs = plt.subplots(1+no_images, 4, figsize=(20, (no_images+1)*5))
    axs[0, 0].plot(tr_err, label='tr_err')
    axs[0, 0].plot(ts_err, label='ts_err')
    axs[0, 0].legend()
    axs[0, 1].plot(tr_acc, label='tr_err')
    axs[0, 1].plot(ts_acc, label='ts_err')
    axs[0, 1].legend()
    axs[0, 2].axis('off')
    axs[0, 3].axis('off')
    for img_no in range(6):
        im = img[img_no].cpu().detach().numpy().transpose(1, 2, 0)*0.5 + 0.5
        axs[img_no+1, 0].imshow(im)
        for i in range(3):
            att_out_img = att_out[img_no, i+1].cpu().detach().numpy()
            axs[img_no+1, i+1].imshow(att_out_img)

    plt.savefig('models/' + save_location + '/Results/training_data.png')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.01)

def train(model, att_flag=False):
    net = model.to(device)
    tr_err, ts_err = [], []
    tr_acc, ts_acc = [], []
    print('Epoch Start')
    for epoch in tqdm(range(EPOCH)):
        errs, accs = [], []
        net.train()

        for i, data in enumerate(trainloader):

            net.optim.zero_grad()
            x, y = data['images'],data['labels']
            x = x.to(device)
            y = y.to(device)
            p, q = net.forward(x)
            loss = -torch.mean(y*torch.log(p+1e-8) + (1-y)*torch.log(1-p+1e-8))
            loss.backward()
            errs.append(loss.cpu().detach().item())
            pred = torch.round(p)
            accs.append(torch.sum(pred == y).cpu().detach().item()/BATCH_SIZE)
            net.optim.step()
        tr_err.append(np.mean(errs))
        tr_acc.append(np.mean(accs))

        errs, accs = [], []
        net.eval()
        for i, data in enumerate(valloader):
            x, y = data['images'],data['labels']
            x = x.to(device)
            y = y.to(device)
            p, q = net.forward(x)
            loss = -torch.mean(y*torch.log(p+1e-8) + (1-y)*torch.log(1-p+1e-8))
            errs.append(loss.cpu().detach().item())
            pred = torch.round(p)
            accs.append(torch.sum(pred == y).cpu().detach().item()/BATCH_SIZE)
        ts_err.append(np.mean(errs))
        ts_acc.append(np.mean(accs))

        if att_flag == False:
            plot_without_attention(tr_err, ts_err, tr_acc, ts_acc, x[0])
        else:
            plot_with_attention(tr_err, ts_err, tr_acc, ts_acc, x, q)

        print(f'Min train error: {np.min(tr_err)}')
        print(f'Min test error: {np.min(ts_err)}')

model = NetCBAM()
train(model, att_flag=True)


