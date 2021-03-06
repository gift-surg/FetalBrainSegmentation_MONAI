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

import os
import sys
import logging
from datetime import datetime
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch
from ignite.handlers import ModelCheckpoint, EarlyStopping

sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.data import list_data_collate
from monai.transforms import Compose, LoadNiftid, AddChanneld, NormalizeIntensityd, Resized, \
     RandSpatialCropd, RandRotated, RandFlipd, SqueezeDimd, ToTensord
from monai.handlers import CheckpointLoader, \
    StatsHandler, TensorBoardStatsHandler, TensorBoardImageHandler, LrScheduleHandler, MeanDice, stopping_fn_from_metric
from monai.networks.utils import predict_segmentation
from monai.utils import set_determinism
from monai.metrics import compute_meandice
from monai.visualize import plot_2d_or_3d_image

from io_utils import create_data_list
from sliding_window_inference import sliding_window_inference


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
    cuda_device = config_info['device']['cuda_device']
    num_workers = config_info['device']['num_workers']
    # training and validation params
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

    torch.cuda.set_device(cuda_device)
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
    # - apply whitening
    # - resize to (96, 96) in-plane (preserve z-direction)
    # - define 2D patches to be extracted
    # - add data augmentation (random rotation and random flip)
    # - squeeze to 2D
    train_transforms = Compose([
        LoadNiftid(keys=['img', 'seg']),
        AddChanneld(keys=['img', 'seg']),
        NormalizeIntensityd(keys=['img']),
        Resized(keys=['img', 'seg'], spatial_size=[96, 96], interp_order=[1, 0], anti_aliasing=[True, False]),
        RandSpatialCropd(keys=['img', 'seg'], roi_size=[96, 96, 1], random_size=False),
        RandRotated(keys=['img', 'seg'], degrees=90, prob=0.2, spatial_axes=[0, 1], interp_order=[1, 0], reshape=False),
        RandFlipd(keys=['img', 'seg'], spatial_axis=[0, 1]),
        SqueezeDimd(keys=['img', 'seg'], dim=-1),
        ToTensord(keys=['img', 'seg'])
    ])
    # create a training data loader
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0,
    #                                    num_workers=num_workers)
    train_ds = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=persistent_cache)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size_train,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    # check_train_data = monai.utils.misc.first(train_loader)
    # print("Training data tensor shapes")
    # print(check_train_data['img'].shape, check_train_data['seg'].shape)

    # data preprocessing for validation:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - apply whitening
    # - resize to (96, 96) in-plane (preserve z-direction)
    if sliding_window_validation:
        val_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            AddChanneld(keys=['img', 'seg']),
            NormalizeIntensityd(keys=['img']),
            Resized(keys=['img', 'seg'], spatial_size=[96, 96], interp_order=[1, 0], anti_aliasing=[True, False]),
            ToTensord(keys=['img', 'seg'])
        ])
        do_shuffle = False
        collate_fn_to_use = None
    else:
        # - add extraction of 2D slices from validation set to emulate how loss is computed at training
        val_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            AddChanneld(keys=['img', 'seg']),
            NormalizeIntensityd(keys=['img']),
            Resized(keys=['img', 'seg'], spatial_size=[96, 96], interp_order=[1, 0], anti_aliasing=[True, False]),
            RandSpatialCropd(keys=['img', 'seg'], roi_size=[96, 96, 1], random_size=False),
            SqueezeDimd(keys=['img', 'seg'], dim=-1),
            ToTensord(keys=['img', 'seg'])
        ])
        do_shuffle = True
        collate_fn_to_use = list_data_collate
    # create a validation data loader
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0,
    #                                    num_workers=num_workers)
    val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size_valid,
                            shuffle=do_shuffle,
                            collate_fn=collate_fn_to_use,
                            num_workers=num_workers)
    # check_valid_data = monai.utils.misc.first(val_loader)
    # print("Validation data tensor shapes")
    # print(check_valid_data['img'].shape, check_valid_data['seg'].shape)

    """
    Network preparation
    """
    # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    loss_function = monai.losses.DiceLoss(do_sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), lr)
    device = torch.cuda.current_device()
    if lr_decay is not None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=lr_decay, last_epoch=-1)

    """
    Set ignite trainer
    """
    # function to manage batch at training
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch['img'], batch['seg']), device, non_blocking)

    trainer = create_supervised_trainer(model=net, optimizer=opt, loss_fn=loss_function,
                                        device=device, non_blocking=False, prepare_batch=prepare_batch)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    if model_to_load is not None:
        checkpoint_handler = CheckpointLoader(load_path=model_to_load, load_dict={'net': net,
                                                                                  'opt': opt,
                                                                                  })
        checkpoint_handler.attach(trainer)
        state = trainer.state_dict()
    else:
        checkpoint_handler = ModelCheckpoint(out_model_dir, 'net', n_saved=max_nr_models_saved, require_empty=False)
        # trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=save_params)
        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                                  handler=checkpoint_handler,
                                  to_save={'net': net, 'opt': opt})

    # StatsHandler prints loss at every iteration and print metrics at every epoch
    train_stats_handler = StatsHandler(name='trainer')
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    writer_train = SummaryWriter(log_dir=os.path.join(out_model_dir, "train"))
    train_tensorboard_stats_handler = TensorBoardStatsHandler(summary_writer=writer_train)
    train_tensorboard_stats_handler.attach(trainer)

    if lr_decay is not None:
        print("Using Exponential LR decay")
        lr_schedule_handler = LrScheduleHandler(lr_scheduler, print_lr=True, name="lr_scheduler", writer=writer_train)
        lr_schedule_handler.attach(trainer)

    """
    Set ignite evaluator to perform validation at training
    """
    # set parameters for validation
    metric_name = 'Mean_Dice'
    # add evaluation metric to the evaluator engine
    val_metrics = {
        "Loss": 1.0 - MeanDice(add_sigmoid=True, to_onehot_y=False),
        "Mean_Dice": MeanDice(add_sigmoid=True, to_onehot_y=False)
    }

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch['img'].to(device), batch['seg'].to(device)
            roi_size = (96, 96, 1)
            seg_probs = sliding_window_inference(val_images, roi_size, batch_size_valid, net)
            return seg_probs, val_labels

    if sliding_window_validation:
        # use sliding window inference at validation
        print("3D evaluator is used")
        net.to(device)
        evaluator = Engine(_sliding_window_processor)
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)
    else:
        # ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
        # user can add output_transform to return other values
        print("2D evaluator is used")
        evaluator = create_supervised_evaluator(model=net, metrics=val_metrics, device=device,
                                                non_blocking=True, prepare_batch=prepare_batch)

    epoch_len = len(train_ds) // train_loader.batch_size
    validation_every_n_iters = validation_every_n_epochs * epoch_len

    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def run_validation(engine):
        evaluator.run(val_loader)

    # add early stopping handler to evaluator
    # early_stopper = EarlyStopping(patience=4,
    #                               score_function=stopping_fn_from_metric(metric_name),
    #                               trainer=trainer)
    # evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name='evaluator',
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    writer_valid = SummaryWriter(log_dir=os.path.join(out_model_dir, "valid"))
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        summary_writer=writer_valid,
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.iteration)  # fetch global iteration number from trainer
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding label and model output in the last batch
    # here we draw the 3D output as GIF format along the depth axis, every 2 validation iterations.
    if val_image_to_tensorboad:
        val_tensorboard_image_handler = TensorBoardImageHandler(
            summary_writer=writer_valid,
            batch_transform=lambda batch: (batch['img'], batch['seg']),
            output_transform=lambda output: predict_segmentation(output[0]),
            global_iter_transform=lambda x: trainer.state.epoch
        )
        evaluator.add_event_handler(
            event_name=Events.ITERATION_COMPLETED(every=1), handler=val_tensorboard_image_handler)

    """
    Run training
    """
    state = trainer.run(train_loader, nr_train_epochs)
    print("Done!")


if __name__ == "__main__":
    main()
