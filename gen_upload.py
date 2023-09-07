import sys
sys.path.append('model')

import os
import imageio
from tqdm import tqdm
import numpy as np
import glob
import torch
import time

from model.TMA import TMA


@torch.no_grad()
def upload_TMA_DSEC(args):
    model = TMA(input_bins=15)
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_ed + '.pth')
    ckpt = torch.load(ckpt_path)
    print('Processing ', ckpt_path)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    voxels = glob.glob(os.path.join(args.test_path, 'test','*','*.npz'))
    voxels.sort()
    time_list = []
    bar = tqdm(voxels, total=len(voxels), ncols=80)
    for f in bar:
        voxel1 = np.load(f)['voxel_prev']
        voxel2 = np.load(f)['voxel_curr']
        city = f.split('/')[-2]
        ind = f.split('/')[-1].split('.')[0].split('_')[-1]
        voxel1 = torch.from_numpy(voxel1)[None].cuda()
        voxel2 = torch.from_numpy(voxel2)[None].cuda()

        start = time.time()
        flow_up = model(voxel1, voxel2)
        end = time.time()
        time_list.append((end-start)*1000)
        flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

        uv = flo * 128.0 + 2**15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)


        test_save = args.test_save
        city = os.path.join(test_save, city)
        if not os.path.exists(city):
            os.makedirs(city)
        path_to_file = os.path.join(city, ind+'.png')
        imageio.imwrite(path_to_file, uv, format='PNG-FI')
    avg_time = sum(time_list)/len(time_list)
    print(f'Time: {avg_time} ms.')  
    print('Done!')

      
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='upload')

    parser.add_argument('--checkpoint_dir', type=str, default='ckpts/')
    parser.add_argument('--checkpoint_ed', type=str, default='')
    #save setting
    parser.add_argument('--test_path', default='')
    parser.add_argument('--test_save', default='upload/')
    args = parser.parse_args()

    upload_TMA_DSEC(args)
