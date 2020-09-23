# Copyright 2020 Marta Bianca Maria Ranzini and contributors
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
import warnings
import sys
from datetime import datetime
import yaml
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import BCEWithLogitsLoss
from ignite.metrics import Accuracy
from ignite.engine import Events

import monai
from monai.data import list_data_collate
from monai.utils import set_determinism, is_scalar
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    CheckpointLoader
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
    RandSpatialCropd,
    RandRotated,
    RandFlipd,
    SqueezeDimd,
    ToTensord,
)

from io_utils import create_data_list
from sliding_window_inference import sliding_window_inference
from custom_ignite_engines import create_supervised_trainer_with_clipping, create_evaluator_with_sliding_window
from custom_unet import CustomUNet
from custom_losses import DiceAndBinaryXentLoss, TverskyLoss_noSmooth, DiceLossExtended
from custom_metrics import MeanDiceAndBinaryXentMetric, BinaryXentMetric, TverskyMetric
from custom_transform import ConverToOneHotd, MinimumPadd
from custom_inferer import SlidingWindowInferer2D
from custom_trainer import SupervisedTrainerClipping
from custom_handlers import MyTensorBoardImageHandler

DEFAULT_KEY_VAL_FORMAT = '{}: {:.4f} '
DEFAULT_TAG = 'Loss'


def my_iteration_print_logger(engine):
    """Execute iteration log operation based on Ignite engine.state data.
    Print the values from ignite state.logs dict.
    Default behavior is to print loss from output[1], skip if output[1] is not loss.

    Args:
        engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

    """
    key_var_format = DEFAULT_KEY_VAL_FORMAT
    tag_name = DEFAULT_TAG
    loss = engine.state.output
    if loss is None:
        return  # no printing if the output is empty

    out_str = ''
    if isinstance(loss, dict):  # print dictionary items
        for name in sorted(loss):
            value = loss[name]
            if not is_scalar(value):
                warnings.warn('ignoring non-scalar output in StatsHandler,'
                              ' make sure `output_transform(engine.state.output)` returns'
                              ' a scalar or dictionary of key and scalar pairs to avoid this warning.'
                              ' {}:{}'.format(name, type(value)))
                continue  # not printing multi dimensional output
            out_str += key_var_format.format(name, value.item() if torch.is_tensor(value) else value)
    else:
        if is_scalar(loss):  # not printing multi dimensional output
            out_str += key_var_format.format(tag_name, loss.item() if torch.is_tensor(loss) else loss)
        else:
            warnings.warn('ignoring non-scalar output in StatsHandler,'
                          ' make sure `output_transform(engine.state.output)` returns'
                          ' a scalar or a dictionary of key and scalar pairs to avoid this warning.'
                          ' {}'.format(type(loss)))

    if not out_str:
        return  # no value to print

    num_iterations = engine.state.epoch_length
    # current_iteration = (engine.state.iteration - 1) % num_iterations + 1
    current_iteration = engine.state.iteration
    current_epoch = engine.state.epoch
    num_epochs = engine.state.max_epochs

    base_str = "Epoch: {}/{}, Iter: {} --".format(
        current_epoch,
        num_epochs,
        current_iteration,
        num_iterations)

    engine.logger.info(' '.join([base_str, out_str]))


def main():
    """
    Basic UNet as implemented in MONAI for Fetal Brain Segmentation, but using
    ignite to manage training and validation loop and checkpointing
    :return:
    """

    """
    Read input and configuration parameters
    """
    parser = argparse.ArgumentParser(description='Run basic UNet with MONAI - Ignite version.')
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
    inplane_size = config_info["training"]["inplane_size"]
    nnunet_preprocessing = config_info["training"]["nnunet_preprocessing"]
    loss_type = config_info['training']['loss_type']
    batch_size_train = config_info['training']['batch_size_train']
    batch_size_valid = config_info['training']['batch_size_valid']
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
        # # set torch only seed
        # torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

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
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - convert segmentation to OneHot
    # - apply whitening
    # - resize to (96, 96) in-plane (preserve z-direction)
    # - define 2D patches to be extracted
    # - add data augmentation (random rotation and random flip)
    # - squeeze to 2D
    if nnunet_preprocessing:
        train_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            ConverToOneHotd(keys=['seg'], labels=seg_labels),
            AddChanneld(keys=['img']),
            NormalizeIntensityd(keys=['img']),
            MinimumPadd(keys=['img', 'seg'], k=inplane_size + [1]),
            RandSpatialCropd(keys=['img', 'seg'], roi_size=inplane_size + [1], random_size=False),
            RandRotated(keys=['img', 'seg'], range_x=90, range_y=90, prob=0.2, keep_size=True,
                        mode=["bilinear", "nearest"]),
            RandFlipd(keys=['img', 'seg'], spatial_axis=[0, 1]),
            SqueezeDimd(keys=['img', 'seg'], dim=-1),
            ToTensord(keys=['img', 'seg'])
        ])
    else:
        train_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            ConverToOneHotd(keys=['seg'], labels=seg_labels),
            AddChanneld(keys=['img']),
            NormalizeIntensityd(keys=['img']),
            Resized(keys=['img', 'seg'], spatial_size=inplane_size + [-1], mode=["trilinear", "nearest"]),
            RandSpatialCropd(keys=['img', 'seg'], roi_size=inplane_size + [1], random_size=False),
            RandRotated(keys=['img', 'seg'], range_x=90, range_y=90, prob=0.2, keep_size=True,
                        mode=["bilinear", "nearest"]),
            RandFlipd(keys=['img', 'seg'], spatial_axis=[0, 1]),
            SqueezeDimd(keys=['img', 'seg'], dim=-1),
            ToTensord(keys=['img', 'seg'])
        ])

    # create a training data loader
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0,
    #                                    num_workers=num_workers)
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
    # - resize to (96, 96) in-plane (preserve z-direction)
    if sliding_window_validation:
        if nnunet_preprocessing:
            val_transforms = Compose([
                LoadNiftid(keys=['img', 'seg']),
                ConverToOneHotd(keys=['seg'], labels=seg_labels),
                AddChanneld(keys=['img']),
                NormalizeIntensityd(keys=['img']),
                MinimumPadd(keys=['img', 'seg'], k=inplane_size + [1]),
                ToTensord(keys=['img', 'seg'])
            ])
        else:
            val_transforms = Compose([
                LoadNiftid(keys=['img', 'seg']),
                ConverToOneHotd(keys=['seg'], labels=seg_labels),
                AddChanneld(keys=['img']),
                NormalizeIntensityd(keys=['img']),
                Resized(keys=['img', 'seg'], spatial_size=inplane_size + [-1], mode=["trilinear", "nearest"]),
                ToTensord(keys=['img', 'seg'])
            ])
        do_shuffle = False
    else:
        # - add extraction of 2D slices from validation set to emulate how loss is computed at training
        if nnunet_preprocessing:
            val_transforms = Compose([
                LoadNiftid(keys=['img', 'seg']),
                ConverToOneHotd(keys=['seg'], labels=seg_labels),
                AddChanneld(keys=['img']),
                NormalizeIntensityd(keys=['img']),
                MinimumPadd(keys=['img', 'seg'], k=inplane_size + [1]),
                RandSpatialCropd(keys=['img', 'seg'], roi_size=inplane_size + [1], random_size=False),
                SqueezeDimd(keys=['img', 'seg'], dim=-1),
                ToTensord(keys=['img', 'seg'])
            ])
        else:
            val_transforms = Compose([
                LoadNiftid(keys=['img', 'seg']),
                ConverToOneHotd(keys=['seg'], labels=seg_labels),
                AddChanneld(keys=['img']),
                NormalizeIntensityd(keys=['img']),
                Resized(keys=['img', 'seg'], spatial_size=inplane_size + [-1], mode=["trilinear", "nearest"]),
                RandSpatialCropd(keys=['img', 'seg'], roi_size=inplane_size + [1], random_size=False),
                SqueezeDimd(keys=['img', 'seg'], dim=-1),
                ToTensord(keys=['img', 'seg'])
            ])
        do_shuffle = True
    # create a validation data loader
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0,
    #                                    num_workers=num_workers)
    val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
    val_loader = monai.data.DataLoader(val_ds,
                                       batch_size=batch_size_valid,
                                       shuffle=do_shuffle,
                                       num_workers=num_workers)
    check_valid_data = monai.utils.misc.first(val_loader)
    print("Validation data tensor shapes")
    print(check_valid_data['img'].shape, check_valid_data['seg'].shape)

    """
    Network preparation
    """
    current_device = torch.device("cuda:0")
    # # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=nr_out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(current_device)
    # net = CustomUNet()
    print("Model summary:")
    summary(net, input_data=[1] + inplane_size)

    squared_pred = False
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = True
        do_softmax = False
    if loss_type == "Dice":
        smooth_num = 1e-5
        smooth_den = smooth_num
        # loss_function = monai.losses.DiceLoss(sigmoid=do_sigmoid, softmax=do_softmax)
        loss_function = DiceLossExtended(sigmoid=do_sigmoid, softmax=do_softmax,
                                         smooth_num=smooth_num, smooth_den=smooth_den, squared_pred=squared_pred)
        print(f"[LOSS] Using DiceLossExtended with smooth = {smooth_num}, "
              f"do_sigmoid={do_sigmoid}, do_softmax={do_softmax}, squared_pred={squared_pred}")
    elif loss_type == "Xent":
        loss_function = BCEWithLogitsLoss(reduction="mean")
        print("[LOSS] Using BCEWithLogitsLoss")
    elif loss_type == "Dice_nosmooth":
        smooth_num = 0.0
        smooth_den = 1e-5
        # loss_function = DiceLoss_noSmooth(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        loss_function = DiceLossExtended(sigmoid=do_sigmoid, softmax=do_softmax,
                                         smooth_num=smooth_num, smooth_den=smooth_den, squared_pred=squared_pred)
        print(f"[LOSS] Using DiceLossExtended, Dice with smooth_num={smooth_num} and smooth_den={smooth_den},"
              f"do_sigmoid={do_sigmoid}, do_softmax={do_softmax}, squared_pred={squared_pred}")
    elif loss_type == "Batch_Dice":
        smooth_num = 1e-5
        smooth_den = smooth_num
        loss_function = DiceLossExtended(sigmoid=do_sigmoid, softmax=do_softmax,
                                         smooth_num=smooth_num, smooth_den=smooth_den, squared_pred=squared_pred,
                                         batch_version=True)
        print(f"[LOSS] Using DiceLossExtended - BATCH VERSION, "
              f"Dice with {smooth_num} at numerator and {smooth_den} at denominator, "
              f"do_sigmoid={do_sigmoid}, do_softmax={do_softmax}, squared_pred={squared_pred}")
    elif loss_type == "Tversky":
        loss_function = monai.losses.TverskyLoss(sigmoid=do_sigmoid, softmax=do_softmax)
        print(f"[LOSS] Using monai.losses.TverskyLoss with do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Tversky_nosmooth":
        loss_function = TverskyLoss_noSmooth(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Tversky with no smooth at numerator with do_sigmoid={do_sigmoid}, "
              f"do_softmax={do_softmax}")
    elif loss_type == "Dice_Xent":
        smooth_num = 1e-5
        smooth_den = 1e-5
        batch_version = True
        loss_function = DiceAndBinaryXentLoss(sigmoid=do_sigmoid, softmax=do_softmax,
                                              smooth_num=smooth_num, smooth_den=smooth_den, batch_version=batch_version)
        print(f"[LOSS] Using Custom loss, Dice + Xent with do_sigmoid={do_sigmoid}, do_softmax={do_softmax},"
              f"Dice with {smooth_num} at numerator and {smooth_den} at denominator, "
              f" squared_pred={squared_pred} and batch_version={batch_version}")
    else:
        raise IOError("Unrecognized loss type")

    if optimiser_choice in ("Adam", "adam"):
        opt = torch.optim.Adam(net.parameters(), lr)
        print("[OPTIMISER] Using Adam")
    elif optimiser_choice in ("SGD", "sgd"):
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        print("[OPTIMISER] Using Vanilla SGD")
    elif optimiser_choice in ("SGDMomentum", "sgdmomentum", "SGD_Momentum"):
        momentum = opt_momentum if not None else 0.9
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        print("[OPTIMISER]Using SDG with momentum = {}".format(momentum))
    else:
        print("[OPTIMISER] WARNING: Invalid optimiser choice, using Adam by default")
        opt = torch.optim.Adam(net.parameters(), lr)

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

    if sliding_window_validation:
        print("3D evaluator is used")
        inferer = SlidingWindowInferer2D(roi_size=inplane_size + [1], sw_batch_size=4, overlap=0.0)
    else:
        print("2D evaluator is used")
        inferer = SimpleInferer()

    evaluator = SupervisedEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        inferer=inferer,
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

    writer_train = SummaryWriter(log_dir=os.path.join(out_model_dir, "train"))
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=validation_every_n_iters, epoch_level=False),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(summary_writer=writer_train,
                                log_dir=os.path.join(out_model_dir, "train"), tag_name="Loss",
                                output_transform=lambda x: x["loss"],
                                global_epoch_transform=lambda x: trainer.state.iteration),
        CheckpointSaver(save_dir=out_model_dir, save_dict={"net": net, "opt": opt},
                        save_final=True, save_key_metric=True, key_metric_name='Mean_dice', key_metric_n_saved=1,
                        save_interval=2, epoch_level=True,
                        n_saved=max_nr_models_saved),
    ]
    if model_to_load is not None:
        train_handlers.append(CheckpointLoader(load_path=model_to_load, load_dict={"net": net, "opt": opt}))

    if lr_scheduler is not None:
        print("Using Exponential LR decay")
        train_handlers.append(LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True))

    trainer = SupervisedTrainerClipping(
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
        key_train_metric={"Mean_dice": MeanDice(include_background=True,
                                                output_transform=lambda x: (x["pred"], x["label"]))},
        train_handlers=train_handlers,
        grad_clipping=clipping
    )
    print("Using gradient norm clipping at max = {}".format(clipping))

    # train_tensorboard_image_handler = MyTensorBoardImageHandler(
    #     summary_writer=writer_train,
    #     batch_transform=lambda batch: (batch['img'], batch['seg']),
    #     output_transform=lambda output: None,
    #     global_iter_transform=lambda x: trainer.state.iteration,
    #     index=range(0, 6)
    # )
    # trainer.add_event_handler(
    #     event_name=Events.ITERATION_COMPLETED(every=17088), handler=train_tensorboard_image_handler)

    # TODO: add plotting of gradients
    # get the network gradients and add them to tensorboard
    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def get_network_gradient(engine):
        parameters = net.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(2)
        total_norm = 0
        for idx, p in enumerate(parameters):
            # # TODO add name to each layer so gradients can be mapped (use p.name instead of grad_{})
            # writer_train.add_histogram("grad_{}".format(idx), p.grad, engine.state.iteration)
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        # print("Total norm = {}".format(total_norm))
        max_gradient = max(p.grad.data.max() for p in parameters)
        min_gradient = min(p.grad.data.min() for p in parameters)
        writer_train.add_scalar("Total_Gradient_norm", total_norm, engine.state.iteration)
        writer_train.add_scalar("Max_Gradient", max_gradient, engine.state.iteration)
        writer_train.add_scalar("Min_Gradient", min_gradient, engine.state.iteration)

    """
    Run training
    """
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()
