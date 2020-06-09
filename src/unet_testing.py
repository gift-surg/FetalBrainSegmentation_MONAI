# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import copy
import shutil
from glob import glob
import logging
import yaml
import argparse
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader

sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.data import list_data_collate, sliding_window_inference, create_test_image_3d, NiftiSaver
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadNiftid, AddChanneld, NormalizeIntensityd, Resized, SqueezeDimd, ToTensord

from io_utils import create_data_list
from sliding_window_inference import sliding_window_inference

sys.path.append("/mnt/data/mranzini/Code/Demic-v0.1")
from Demic.util.image_process import *


def main():

    """
    Read input and configuration parameters
    """
    parser = argparse.ArgumentParser(description='Run inference with basic UNet with MONAI.')
    parser.add_argument('--config', dest='config', metavar='config', type=str,
                        help='config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)

    # print to log the parameter setups
    print(yaml.dump(config_info))

    # GPU params
    cuda_device = config_info['device']['cuda_device']
    num_workers = config_info['device']['num_workers']
    # inference params
    batch_size_inference = config_info['inference']['batch_size_inference']
    # temporary check as sliding window inference does not accept higher batch size
    assert batch_size_inference == 1
    prob_thr = config_info['inference']['probability_threshold']
    model_to_load = config_info['inference']['model_to_load']
    if not os.path.exists(model_to_load):
        raise IOError('Trained model not found')
    # data params
    data_root = config_info['data']['data_root']
    inference_list = config_info['data']['inference_list']
    # output saving
    out_dir = config_info['output']['out_dir']

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    torch.cuda.set_device(cuda_device)

    """
    Data Preparation
    """
    val_files = create_data_list(data_folder_list=data_root,
                                 subject_list=inference_list,
                                 img_postfix='_Image',
                                 is_inference=True)

    print(len(val_files))
    print(val_files[0])
    print(val_files[-1])

    # data preprocessing for inference:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - apply whitening
    # - NOTE: resizing needs to be applied afterwards, otherwise it cannot be remapped back to original size
    val_transforms = Compose([
        LoadNiftid(keys=['img']),
        AddChanneld(keys=['img']),
        NormalizeIntensityd(keys=['img']),
        ToTensord(keys=['img'])
    ])
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size_inference,
                            num_workers=num_workers)

    """
    Network preparation
    """
    device = torch.cuda.current_device()
    # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    net.load_state_dict(torch.load(model_to_load))
    net.eval()

    """
    Run inference
    """
    with torch.no_grad():
        saver = NiftiSaver(output_dir=out_dir)
        for val_data in val_loader:
            val_images = val_data['img'].to(device)
            orig_size = list(val_images.shape)
            resized_size = copy.deepcopy(orig_size)
            resized_size[2] = 96
            resized_size[3] = 96
            val_images_resize = torch.nn.functional.interpolate(val_images, size=resized_size[2:], mode='trilinear')
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96, 1)
            val_outputs = sliding_window_inference(val_images_resize, roi_size, batch_size_inference, net)
            val_outputs = (val_outputs.sigmoid() >= prob_thr).float()
            val_outputs_resized = torch.nn.functional.interpolate(val_outputs, size=orig_size[2:], mode='nearest')
            # add post-processing
            val_outputs_resized = val_outputs_resized.detach().cpu().numpy()
            strt = ndimage.generate_binary_structure(3, 2)
            post = padded_binary_closing(np.squeeze(val_outputs_resized), strt)
            post = get_largest_component(post)
            val_outputs_resized = val_outputs_resized * post
            # out = np.zeros(img.shape[:-1], np.uint8)
            # out = set_ND_volume_roi_with_bounding_box_range(out, bb_min, bb_max, out_roi)

            saver.save_batch(val_outputs_resized, {'filename_or_obj': val_data['img.filename_or_obj'],
                                                    'affine': val_data['img.affine']})

if __name__ == '__main__':
    main()
