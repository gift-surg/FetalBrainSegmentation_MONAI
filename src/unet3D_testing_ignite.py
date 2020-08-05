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

import logging
import os
import shutil
import sys
from datetime import datetime
import yaml
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torchsummary import summary
from torch.nn.modules.loss import BCEWithLogitsLoss
from ignite.metrics import Accuracy

import monai
from monai.data import list_data_collate
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, MeanDice, SegmentationSaver, StatsHandler
from monai.inferers import SlidingWindowInferer
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

from io_utils import create_data_list
from custom_transform import ConverToOneHotd, MinimumPadd
from custom_losses import DiceAndBinaryXentLoss, DiceLoss_noSmooth
from custom_networks import CustomUNet25, ShallowUNet
from custom_inferer import SlidingWindowInfererWithResize


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
    patch_size = config_info["inference"]["patch_size"]
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
    val_loader = monai.data.DataLoader(val_ds,
                                       batch_size=batch_size_inference,
                                       num_workers=num_workers)

    def prepare_batch(batchdata):
        assert isinstance(batchdata, dict), "prepare_batch expects dictionary input data."
        return (
            (batchdata['img'], batchdata['seg'])
            if 'seg' in batchdata
            else (batchdata['img'], None)
        )

    # create UNet, DiceLoss and Adam optimizer
    current_device = torch.device("cuda:0")
    # net = monai.networks.nets.UNet(
    #     dimensions=3,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)
    # net = CustomUNet25().to(current_device)
    net = ShallowUNet().to(current_device)

    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            # AsDiscreted(keys="pred", threshold_values=True),
            # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
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
        inferer=SlidingWindowInfererWithResize(roi_size=patch_size, sw_batch_size=1, overlap=0.5),
        prepare_batch=prepare_batch,
        post_transform=val_post_transforms,
        # key_val_metric={
        #     "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        # },
        # additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        val_handlers=val_handlers,
    )

    evaluator.run()


if __name__ == "__main__":
    main()