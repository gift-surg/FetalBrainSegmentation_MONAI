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
from monai.utils import set_determinism
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    NormalizeIntensityd,
    Resized,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotated,
    RandFlipd,
    ToTensord,
)

from io_utils import create_data_list
from custom_transform import ConverToOneHotd, MinimumPadd
from custom_losses import DiceAndBinaryXentLoss, DiceLoss_noSmooth
from custom_networks import CustomUNet25, ShallowUNet
from logging_utils import my_iteration_print_logger


def main():
    """
    3D UNet to predict heatmaps of brain localisation in support to fetal brain segmentation
    :return:
    """

    """
    Read input and configuration parameters
    """
    parser = argparse.ArgumentParser(description='Run 3D UNet with MONAI - Ignite version.')
    parser.add_argument('--config', dest='config', metavar='config', type=str,
                        help='config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)

    # print to log the parameter setups
    print(yaml.dump(config_info))

    # GPU params
    num_workers = config_info['device']['num_workers']
    # training and validation params
    if 'seg_labels' in config_info['training'].keys():
        seg_labels = config_info['training']['seg_labels']
        print(seg_labels)
    else:
        seg_labels = [1]
    nr_out_channels = len(seg_labels)
    print("Considering the following {} labels in the segmentation: {}".format(nr_out_channels, seg_labels))
    loss_type = config_info['training']['loss_type']
    batch_size_train = config_info['training']['batch_size_train']
    batch_size_valid = config_info['training']['batch_size_valid']
    inplane_size = config_info["training"]["inplane_size"]
    outplane_size = config_info["training"]["outplane_size"]
    lr = float(config_info['training']['lr'])
    lr_decay = config_info['training']['lr_decay']
    if lr_decay is not None:
        lr_decay = float(lr_decay)
    nr_train_epochs = config_info['training']['nr_train_epochs']
    validation_every_n_epochs = config_info['training']['validation_every_n_epochs']
    sliding_window_validation = config_info['training']['sliding_window_validation']
    if 'model_to_load' in config_info['training'].keys():
        model_to_load = config_info['training']['model_to_load']
        if not os.path.exists(model_to_load):
            raise BlockingIOError("cannot find model: {}".format(model_to_load))
    else:
        model_to_load = None
    if 'manual_seed' in config_info['training'].keys():
        seed = config_info['training']['manual_seed']
    else:
        seed = None
    # optimiser params
    optimiser_choice = config_info['training']['optimiser_choice']
    opt_momentum = config_info['training']['opt_momentum']
    clipping = config_info['training']['gradient_clipping']
    # data params
    data_root = config_info['data']['data_root']
    training_list = config_info['data']['training_list']
    validation_list = config_info['data']['validation_list']
    # model saving
    out_model_dir = os.path.join(config_info['output']['out_model_dir'],
                                 datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' +
                                 config_info['output']['output_subfix'])
    print("Saving to directory ", out_model_dir)
    if 'cache_dir' in config_info['output'].keys():
        out_cache_dir = config_info['output']['cache_dir']
    else:
        out_cache_dir = os.path.join(out_model_dir, 'persistent_cache')
    max_nr_models_saved = config_info['output']['max_nr_models_saved']
    val_image_to_tensorboad = config_info['output']['val_image_to_tensorboad']

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print("\n#### GPU INFORMATION ###")
    print(f"Using device number: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name()}")
    print(f"Device available: {torch.cuda.is_available()}\n")
    if seed is not None:
        # set manual seed if required (both numpy and torch)
        set_determinism(seed=seed)

    """
    Data Preparation
    """
    # create cache directory to store results for Persistent Dataset
    persistent_cache: Path = Path(out_cache_dir)
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # create training and validation data lists
    train_files = create_data_list(data_folder_list=data_root,
                                   subject_list=training_list,
                                   img_postfix='_Image',
                                   label_postfix='_Label')

    print(len(train_files))
    print(train_files[0])
    print(train_files[-1])

    val_files = create_data_list(data_folder_list=data_root,
                                 subject_list=validation_list,
                                 img_postfix='_Image',
                                 label_postfix='_Label')
    print(len(val_files))
    print(val_files[0])
    print(val_files[-1])

    # data preprocessing for training:
    # data preprocessing for training:
    patch_size = inplane_size + [outplane_size]
    train_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            ConverToOneHotd(keys=["seg"], labels=seg_labels),
            AddChanneld(keys=["img"]),
            NormalizeIntensityd(keys=["img"]),
            MinimumPadd(keys=["img", "seg"], k=(-1, -1, outplane_size)),
            Resized(keys=["img", "seg"], spatial_size=inplane_size + [-1]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=patch_size, pos=1, neg=1, num_samples=2
            ),
            RandRotated(keys=["img", "seg"], range_x=90, range_y=90, prob=0.5, keep_size=True,
                        mode=["bilinear", "nearest"]),
            RandFlipd(keys=["img", "seg"], spatial_axis=[0, 1]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    # create training data loader
    train_ds = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=persistent_cache)
    train_loader = monai.data.DataLoader(train_ds,
                                         batch_size=batch_size_train,
                                         shuffle=True, num_workers=num_workers,
                                         pin_memory=torch.cuda.is_available())
    check_train_data = monai.utils.misc.first(train_loader)
    print("Training data tensor shapes")
    print(check_train_data['img'].shape, check_train_data['seg'].shape)

    # data preprocessing for validation:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - apply whitening
    val_transforms = Compose(
        [
            LoadNiftid(keys=['img', 'seg']),
            ConverToOneHotd(keys=['seg'], labels=seg_labels),
            AddChanneld(keys=['img']),
            NormalizeIntensityd(keys=['img']),
            MinimumPadd(keys=["img", "seg"], k=(-1, -1, outplane_size)),
            Resized(keys=["img", "seg"], spatial_size=inplane_size + [-1]),
            ToTensord(keys=['img', 'seg'])
        ]
    )

    # create a validation data loader
    val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
    val_loader = monai.data.DataLoader(val_ds,
                                       batch_size=batch_size_valid,
                                       shuffle=False,
                                       num_workers=num_workers)
    check_valid_data = monai.utils.misc.first(val_loader)
    print("Validation data tensor shapes")
    print(check_valid_data['img'].shape, check_valid_data['seg'].shape)

    """
    Network preparation
    """
    current_device = torch.device("cuda:0")
    # create UNet, DiceLoss and Adam optimizer
    # net = monai.networks.nets.UNet(
    #     dimensions=3,
    #     in_channels=1,
    #     out_channels=nr_out_channels,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(current_device)
    # net = CustomUNet25().to(current_device)
    net = ShallowUNet().to(current_device)
    print("Model summary:")
    summary(net, input_data=[1] + patch_size)

    smooth = None
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = False
        do_softmax = True
    if loss_type == "Dice":
        loss_function = monai.losses.DiceLoss(sigmoid=do_sigmoid, softmax=do_softmax)
        smooth = 1e-5
        print(f"[LOSS] Using monai.losses.DiceLoss with smooth = {smooth}, do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Xent":
        loss_function = BCEWithLogitsLoss(reduction="mean")
        print("[LOSS] Using BCEWithLogitsLoss")
    elif loss_type == "Dice_nosmooth":
        loss_function = DiceLoss_noSmooth(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Dice with no smooth at numerator, do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Dice_Xent":
        loss_function = DiceAndBinaryXentLoss(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Dice + Xent with do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    else:
        raise IOError("Unrecognized loss type")

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = None
    if lr_decay is not None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=lr_decay, last_epoch=-1)

    def prepare_batch(batchdata):
        assert isinstance(batchdata, dict), "prepare_batch expects dictionary input data."
        return (
            (batchdata['img'], batchdata['seg'])
            if 'seg' in batchdata
            else (batchdata['img'], None)
        )

    """
    Set ignite evaluator to perform validation at training
    """
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=os.path.join(out_model_dir, "valid"), output_transform=lambda x: None,
                                global_epoch_transform=lambda x: trainer.state.iteration),
        CheckpointSaver(save_dir=out_model_dir, save_dict={"valid": net}, save_key_metric=True),
    ]
    if val_image_to_tensorboad:
        val_handlers.append(TensorBoardImageHandler(log_dir=os.path.join(out_model_dir, "valid"),
                                                    batch_transform=lambda x: (x["img"], x["seg"]),
                                                    output_transform=lambda x: x["pred"], interval=2))

    evaluator = SupervisedEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        inferer=SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5),
        post_transform=val_post_transforms,
        key_val_metric={
            "Mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        },
        additional_metrics={
            "Loss": 1.0 - MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        },
        val_handlers=val_handlers
    )

    """
    Set ignite trainer 
    """
    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
        ]
    )

    epoch_len = len(train_ds) // train_loader.batch_size
    validation_every_n_iters = validation_every_n_epochs * epoch_len

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=validation_every_n_iters, epoch_level=False),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(log_dir=os.path.join(out_model_dir, "train"), tag_name="Loss",
                                output_transform=lambda x: x["loss"],
                                global_epoch_transform=lambda x: trainer.state.iteration),
        CheckpointSaver(save_dir=out_model_dir, save_dict={"net": net, "opt": opt},
                        save_final=True, save_key_metric=True, key_metric_name='Mean_dice', key_metric_n_saved=1,
                        save_interval=2, epoch_level=True,
                        n_saved=max_nr_models_saved),
    ]

    if lr_scheduler is not None:
        train_handlers.append(LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True))

    trainer = SupervisedTrainer(
        device=current_device,
        max_epochs=nr_train_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss_function,
        prepare_batch=prepare_batch,
        inferer=SimpleInferer(),
        amp=False,
        post_transform=train_post_transforms,
        key_train_metric={"Mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))},
        train_handlers=train_handlers
    )

    trainer.run()


if __name__ == "__main__":
    main()