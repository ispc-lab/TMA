from tqdm import tqdm
import numpy as np
import torch
import os
from datasets.DSEC_split_loader import DSECsplit


@torch.no_grad()                   
def validate_DSEC(model):
    model.eval()
    val_dataset = DSECsplit('test')
    
    epe_list = []
    out_list = []

    bar = tqdm(enumerate(val_dataset),total=len(val_dataset), ncols=60)
    bar.set_description('Test')
    for index, (voxel1, voxel2, flow_map, valid2D) in bar:
        voxel1 = voxel1[None].cuda()
        voxel2 = voxel2[None].cuda() 
        flow_pred = model(voxel1, voxel2)[0].cpu()#[1,2,H,W]

        epe = torch.sum((flow_pred- flow_map)**2, dim=0).sqrt()#[H,W]
        mag = torch.sum(flow_map**2, dim=0).sqrt()#[H,W]

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid2D.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    
    print("Validation DSEC-TEST: %f, %f" % (epe, f1))
    return {'dsec-epe': epe, 'dsec-f1': f1}




