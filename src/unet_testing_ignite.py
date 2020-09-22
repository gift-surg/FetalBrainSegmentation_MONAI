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
import yaml
import argparse
import copy
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine
from torch.utils.data import DataLoader

# sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.data import list_data_collate
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, MeanDice, SegmentationSaver, StatsHandler
from monai.transforms import (
    Activationsd,
    AddChanneld,
    NormalizeIntensityd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadNiftid,
    ToTensord,
)
from monai.networks.utils import predict_segmentation
from monai.networks.nets import UNet

from io_utils import create_data_list
from custom_inferer import SlidingWindowInferer2DWithResize

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
    num_workers = config_info['device']['num_workers']
    # inference params
    nr_out_channels = config_info['inference']['nr_out_channels']
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
    out_postfix = config_info['output']['out_postfix']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print("\n#### GPU INFORMATION ###")
    print(f"Using device number: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name()}")
    print(f"Device available: {torch.cuda.is_available()}\n")

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

    def prepare_batch(batchdata):
        assert isinstance(batchdata, dict), "prepare_batch expects dictionary input data."
        return (
            (batchdata['img'], batchdata['seg'])
            if 'seg' in batchdata
            else (batchdata['img'], None)
        )

    """
    Network preparation
    """
    current_device = torch.device("cuda:0")
    # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=nr_out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(current_device)

    # define sliding window size and batch size for windows inference
    roi_size = (96, 96, 1)

    """
    Set ignite evaluator to perform inference
    """
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = False
        do_softmax = True
    else:
        raise Exception("incompatible number of output channels")
    print(f"Using sigmoid={do_sigmoid} and softmax={do_softmax} as final activation")
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=do_sigmoid, softmax=do_softmax),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=model_to_load, load_dict={"net": net}),
        SegmentationSaver(
            output_dir=out_dir, output_ext='.nii.gz', output_postfix=out_postfix,
            batch_transform=lambda batch: batch["img_meta_dict"],
            output_transform=lambda output: output["pred"],
        ),
    ]

    evaluator = SupervisedEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        inferer=SlidingWindowInferer2DWithResize(roi_size=roi_size, sw_batch_size=4, overlap=0.0),
        post_transform=val_post_transforms,
        # key_val_metric={
        #     "Mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        # },
        # additional_metrics={
        #     "Loss": 1.0 - MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        # },
        val_handlers=val_handlers
    )

    """
    Run inference
    """
    evaluator.run()
    print("Done!")


if __name__ == '__main__':
    main()
