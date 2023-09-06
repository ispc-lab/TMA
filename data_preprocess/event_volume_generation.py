import h5py
import hdf5plugin
from tqdm import tqdm
import numpy as np
import os
import imageio
imageio.plugins.freeimage.download()
from glob import glob

# from taskmanager import TaskManager
from slicer import EventSlicer



def events_to_voxel_grid(events, num_bins, height, width, normalize=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [x, y, timestamp, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 2] = (num_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    ts = events[:, 2]
    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if normalize:
        mask = np.nonzero(voxel_grid)
        if mask[0].size > 0:
            mean, stddev = voxel_grid[mask].mean(), voxel_grid[mask].std()
            if stddev > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / stddev

    return voxel_grid

def rectify_events(x: np.ndarray, y: np.ndarray, rectify_map):
    # From distorted to undistorted
    assert rectify_map.shape == (480, 640, 2)
    assert x.max() < 640
    assert y.max() < 480
    return rectify_map[y, x]

def transform_event_and_write(events_prev, events_curr, flow_16bit, idx, output_dir):

    voxel_prev = events_to_voxel_grid(events_prev, 15, 480, 640)
    voxel_curr = events_to_voxel_grid(events_curr, 15, 480, 640)

    output_name = os.path.join(output_dir, 'seq_{:06d}'.format(idx))

    #save voxel grids
    np.savez(output_name, voxel_prev=voxel_prev, voxel_curr=voxel_curr)
    #save gt flow
    np.save(output_name + '_flow', flow_16bit)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_path', type=str, default='DSEC/train_events')
    parser.add_argument('--gt_path', type=str, default='DSEC/train_optical_flow')
    parser.add_argument('--dst', type=str, default='')
    args = parser.parse_args()
    event_path = args.event_path
    gt_path = args.gt_path
    dst = args.dst
   
    scenes = os.listdir(gt_path) 

    for scene in scenes:
        event_f = os.path.join(event_path, scene, 'events/left', 'events.h5')
        rect_f = os.path.join(event_path, scene, 'events/left', 'rectify_map.h5')

        #event h5 and rectify map h5
        h5f = h5py.File(event_f, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rect_f, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]
        
        #gt timestamps and gt optical flow
        timestamp_f = os.path.join(gt_path, scene, 'flow', 'forward_timestamps.txt')
        timestamps = np.genfromtxt(timestamp_f, delimiter=',')
        gt_flow_f = sorted(glob(os.path.join(gt_path, scene, 'flow', 'forward', '*.png')))
        assert len(timestamps) == len(gt_flow_f)

        #output dir
        output_dir = os.path.join(dst, scene)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # with TaskManager(total=len(timestamps), processes=4, queue_size=4, use_pbar=True, pbar_desc=f"Processing {scene}") as tm:
            
        for idx in tqdm(range(len(timestamps)), ncols=60):
            t_curr = timestamps[idx, 0]
            t_next = timestamps[idx, 1]
            dt = 100 * 1000 #us 
            t_prev = t_curr - dt

            #current event volume
            events_curr = slicer.get_events(t_curr, t_next)
            p = events_curr['p']
            t = events_curr['t']
            x = events_curr['x']
            y = events_curr['y']
            xy_rect = rectify_events(x, y, rectify_map)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]
            mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)
            x_rect = x_rect[mask]
            y_rect = y_rect[mask]
            p = p[mask]
            t = t[mask]
            events_curr = np.stack([x_rect, y_rect, t, p], axis=1)
            
            #previous event volume
            events_prev = slicer.get_events(t_prev, t_curr)
            p = events_prev['p']
            t = events_prev['t']
            x = events_prev['x']
            y = events_prev['y'] 
            xy_rect = rectify_events(x, y, rectify_map)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]
            mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)
            x_rect = x_rect[mask]
            y_rect = y_rect[mask]
            p = p[mask]
            t = t[mask]    
            events_prev = np.stack([x_rect, y_rect, t, p], axis=1)

            #gt flow
            flow_16bit = imageio.imread(gt_flow_f[idx], format='PNG-FI')

            transform_event_and_write(events_prev, events_curr, flow_16bit, idx, output_dir)

        
        h5f.close()
