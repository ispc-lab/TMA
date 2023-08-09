import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

from datasets.augument import Augumentor

class DSECdataset(data.Dataset):
    def __init__(self, args, augument=True):
        super(DSECdataset, self).__init__()
        self.init_seed = False
        self.files = []
        self.flows = []

        self.root = args.root
        self.augment = augument
        if self.augment:
            self.augmentor = Augumentor(crop_size=[288, 384])
        
        self.files = glob.glob(os.path.join(self.root, 'train', '*', 'seq_*.npz'))
        self.files.sort()
        self.flows = glob.glob(os.path.join(self.root, 'train', '*', 'seq_*.npy'))
        self.flows.sort()

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        voxel_file = np.load(self.files[index])
        voxel1 = voxel_file['voxel_prev'].transpose([1,2,0])
        voxel2 = voxel_file['voxel_curr'].transpose([1,2,0])


        flow_16bit = np.load(self.flows[index])
        flow_map, valid2D = flow_16bit_to_float(flow_16bit)

        voxel1, voxel2, flow_map, valid2D = self.augmentor(voxel1, voxel2, flow_map, valid2D)
        
        voxel1 = torch.from_numpy(voxel1).permute(2, 0, 1).float()
        voxel2 = torch.from_numpy(voxel2).permute(2, 0, 1).float()
        flow_map = torch.from_numpy(flow_map).permute(2, 0, 1).float()
        valid2D = torch.from_numpy(valid2D).float()
        return voxel1, voxel2, flow_map, valid2D
    
    def __len__(self):
        return len(self.files)
    
def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def make_data_loader(args, batch_size, num_workers):
    dset = DSECdataset(args)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader
