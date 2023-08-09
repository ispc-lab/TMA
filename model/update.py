import torch
import torch.nn as nn
import torch.nn.functional as F



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class FlowHead(nn.Module):
    def __init__(self, inchannel, hidden_channel=128):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, hidden_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        flow = self.conv2(self.relu(self.conv1(x)))
        return flow


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, split=5):
        super(UpdateBlock, self).__init__()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim + 128 * split)
        self.pred = FlowHead(hidden_dim, hidden_channel=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 8*8*9, 1, padding=0))

    def forward(self, net, inp, mf):
        inp = torch.cat([inp, mf], dim=1)

        net = self.gru(net, inp)
        df = self.pred(net)
        mask = .25 * self.mask(net)
    
        return net, df, mask