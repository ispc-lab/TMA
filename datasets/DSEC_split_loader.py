import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob

from augument import Augumentor

class DSECsplit(data.Dataset):
    def __init__(self, phase):
        super(DSECsplit, self).__init__()
        self.init_seed = False
        self.phase = phase
        self.files = []
        self.flows = []

        ### Please change the root to satisfy your data saving setting.
        if phase == 'train':
            self.root = 'DSEC_split/train'
            self.augmentor = Augumentor(crop_size=[288, 384])
        else:
            self.root = 'DSEC_split/test'


        self.files = glob.glob(os.path.join(self.root, '*', 'seq_*.npz'))
        self.files.sort()
        self.flows = glob.glob(os.path.join(self.root, '*', 'seq_*.npy'))
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
        if self.phase == 'train':
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


def make_data_loader(phase, batch_size, num_workers):
    dset = DSECsplit(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader

if __name__ == '__main__':

    dset = DSECsplit('test')
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)
