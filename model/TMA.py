import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ExtractorF, ExtractorC
from corr import CorrBlock
from aggregate import MotionFeatureEncoder, MPA
from update import UpdateBlock
from util import coords_grid


class TMA(nn.Module):
    def __init__(self, input_bins=15):
        super(TMA, self).__init__()

        f_channel = 128
        self.split = 5
        self.corr_level = 1
        self.corr_radius = 3

        self.fnet = ExtractorF(input_channel=input_bins//self.split, outchannel=f_channel, norm='IN')
        self.cnet = ExtractorC(input_channel=input_bins//self.split + input_bins, outchannel=256, norm='BN')

        self.mfe = MotionFeatureEncoder(corr_level=self.corr_level, corr_radius=self.corr_radius)
        self.mpa = MPA(d_model=128)

        self.update = UpdateBlock(hidden_dim=128, split=self.split)

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)


    def forward(self, x1, x2, iters=6):
        b,_,h,w = x2.shape

        #Feature maps [f_0 :: f_i :: f_g]
        voxels = x2.chunk(self.split, dim=1)
        voxelref = x1.chunk(self.split, dim=1)[-1]
        voxels = (voxelref,) + voxels #[group+1] elements
        fmaps = self.fnet(voxels)#Tuple(f0, f1, ..., f_g)

        # Context map [net, inp]
        cmap = self.cnet(torch.cat(voxels, dim=1))
        net, inp = torch.split(cmap, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords0 = coords_grid(b, h//8, w//8, device=cmap.device)
        coords1 = coords_grid(b, h//8, w//8, device=cmap.device)

        #MidCorr
        corr_fn_list = []
        for i in range(self.split):
            corr_fn = CorrBlock(fmaps[0], fmaps[i+1], num_levels=self.corr_level, radius=self.corr_radius) #[c01,c02,...,c05]
            corr_fn_list.append(corr_fn)

        flow_predictions = []
        for iter in range(iters):

            coords1 = coords1.detach()
            flow = coords1 - coords0

            corr_map_list = []
            du = flow/self.split 
            for i in range(self.split):
                coords = (coords0 + du*(i+1)).detach()
                corr_map = corr_fn_list[i](coords)
                corr_map_list.append(corr_map)

            corr_maps = torch.cat(corr_map_list, dim=0) 

            mfs = self.mfe(torch.cat([flow]*self.split, dim=0), corr_maps)
            mfs = mfs.chunk(self.split, dim=0)
            mfs = self.mpa(mfs)
            mf = torch.cat(mfs, dim=1)
            net, dflow, upmask = self.update(net, inp, mf)
            coords1 = coords1 + dflow
            
            if self.training:
                flow_up = self.upsample_flow(coords1 - coords0, upmask)
                flow_predictions.append(flow_up)

        if self.training:
            return flow_predictions
        else:
            return self.upsample_flow(coords1 - coords0, upmask)

if __name__=='__main__':
    input1 = torch.rand(2,15,288,384)
    input2 = torch.rand(2,15,288,384)
    model = TMA(input_bins=15)
    # model.cuda()
    model.train()
    preds = model(input1, input2)
    print(len(preds))
    model.eval()
    pred = model(input1, input2)
    print(pred.shape)
