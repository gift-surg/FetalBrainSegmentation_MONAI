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

# CODE ADAPTED FROM https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_tutorial.ipynb

import os
import sys
import logging
import yaml
from datetime import datetime
import argparse
from pathlib import Path

import torch
from torch.nn.functional import interpolate

from torch.utils.tensorboard import SummaryWriter
from monai.config import print_config
from monai.data import DataLoader, PersistentDataset
from monai.utils import misc
from monai.engines import SupervisedTrainer
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandRotated,
    RandFlipd,
    SqueezeDimd,
    ToTensord,
    Activationsd,
)

from monai.engines import SupervisedEvaluator
from monai.handlers import (
    LrScheduleHandler,
    StatsHandler,
    CheckpointSaver,
    MeanDice,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    CheckpointLoader
)
from monai.inferers import SlidingWindowInferer, SimpleInferer

from io_utils import create_data_list
from custom_transform import ConverToOneHotd, InPlaneSpacingd
from custom_losses import DiceCELoss, DiceLossExtended, DiceAndBinaryXentLoss
from custom_inferer import SlidingWindowInferer2D


def main():
    """
    Code to train dynUNet for fetal brain segmentation
    """

    """
    Read input and configuration parameters
    """
    parser = argparse.ArgumentParser(description='Run dynUNet with MONAI - Ignite version.')
    parser.add_argument('--config', dest='config', metavar='config', type=str,
                        help='config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)

    # print MONAI config information
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # print to log the parameter setups
    print(yaml.dump(config_info))

    if 'seg_labels' in config_info['training'].keys():
        seg_labels = config_info['training']['seg_labels']
        print(seg_labels)
    else:
        seg_labels = [1]
    nr_out_channels = len(seg_labels)
    print(f"Considering the following {nr_out_channels} labels in the segmentation: {seg_labels}")

    patch_size = config_info["training"]["inplane_size"] + [1]
    print(f"Considering patch size = {patch_size}")

    spacing = config_info["training"]["spacing"]
    print(f"Bringing all images to spacing = {spacing}")

    if 'model_to_load' in config_info['training'].keys():
        model_to_load = config_info['training']['model_to_load']
        if not os.path.exists(model_to_load):
            raise BlockingIOError("cannot find model: {}".format(model_to_load))
        else:
            print(f"Loading model from {model_to_load}")
    else:
        model_to_load = None

    print("\n#### GPU INFORMATION ###")
    print(f"Using device number: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name()}")
    print(f"Device available: {torch.cuda.is_available()}\n")

    """
    Setup data directory
    """
    out_model_dir = os.path.join(config_info['output']['out_model_dir'],
                                 datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' +
                                 config_info['output']['output_subfix'])
    print("Saving to directory ", out_model_dir)
    # create cache directory to store results for Persistent Dataset
    if 'cache_dir' in config_info['output'].keys():
        out_cache_dir = config_info['output']['cache_dir']
    else:
        out_cache_dir = os.path.join(out_model_dir, 'persistent_cache')
    persistent_cache: Path = Path(out_cache_dir)
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # create training and validation data lists
    train_files = create_data_list(data_folder_list=config_info['data']['data_root'],
                                   subject_list=config_info['data']['training_list'],
                                   img_postfix='_Image',
                                   label_postfix='_Label')

    print(len(train_files))
    print(train_files[0])
    print(train_files[-1])

    val_files = create_data_list(data_folder_list=config_info['data']['data_root'],
                                 subject_list=config_info['data']['validation_list'],
                                 img_postfix='_Image',
                                 label_postfix='_Label')
    print(len(val_files))
    print(val_files[0])
    print(val_files[-1])

    """
    Define train and validation transforms
    """
    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            # ConverToOneHotd(keys=["label"], labels=seg_labels),
            AddChanneld(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            InPlaneSpacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size,
                        mode=["constant", "edge"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
            SqueezeDimd(keys=["image", "label"], dim=-1),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=("bilinear", "nearest"),
                align_corners=(True, None),
                prob=0.16,
            ),
            RandRotated(keys=["image", "label"], range_x=90, range_y=90, prob=0.2,
                        keep_size=True, mode=["bilinear", "nearest"],
                        padding_mode=["zeros", "border"]),
            # CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandFlipd(["image", "label"], spatial_axis=[0, 1], prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            # ConverToOneHotd(keys=["label"], labels=seg_labels),
            AddChanneld(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            InPlaneSpacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode=["constant", "edge"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            # CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    """
    Load data 
    """
    train_ds = PersistentDataset(data=train_files, transform=train_transforms,
                                 cache_dir=persistent_cache)
    train_loader = DataLoader(train_ds,
                              batch_size=config_info['training']['batch_size_train'],
                              shuffle=True,
                              num_workers=config_info['device']['num_workers'])
    check_train_data = misc.first(train_loader)
    print("Training data tensor shapes")
    print(check_train_data["image"].shape, check_train_data["label"].shape)

    if config_info['training']['batch_size_valid'] != 1:
        raise Exception("Batch size different from 1 at validation ar currently not supported")
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=config_info['device']['num_workers'])
    check_valid_data = misc.first(val_loader)
    print("Validation data tensor shapes")
    print(check_valid_data["image"].shape, check_valid_data["label"].shape)

    """
    Network preparation and training initialization
    """
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = True
        do_softmax = False

    current_device = torch.device("cuda:0")
    loss_type = config_info['training']['loss_type']
    if loss_type == "dynDiceCELoss":
        batch_version = False
        loss_function = DiceCELoss()
        print(f"[LOSS] Using DiceCELoss with batch_version={batch_version}")
    elif loss_type == "dynDiceCELoss_batch":
        batch_version = True
        loss_function = DiceCELoss(batch_version=batch_version)
        print(f"[LOSS] Using DiceCELoss with batch_version={batch_version}")
    elif loss_type == "Batch_Dice":
        smooth_num = 1e-5
        smooth_den = smooth_num
        batch_version = True
        squared_pred = False
        loss_function = DiceLossExtended(sigmoid=do_sigmoid, softmax=do_softmax,
                                         smooth_num=smooth_num, smooth_den=smooth_den, squared_pred=squared_pred,
                                         batch_version=batch_version)
        print(f"[LOSS] Using DiceLossExtended - BATCH VERSION, "
              f"Dice with {smooth_num} at numerator and {smooth_den} at denominator, "
              f"do_sigmoid={do_sigmoid}, do_softmax={do_softmax}, squared_pred={squared_pred}, "
              f"batch_version={batch_version}")
    elif loss_type == "Dice_Xent":
        smooth_num = 1e-5
        smooth_den = 1e-5
        batch_version = True
        squared_pred = False
        loss_function = DiceAndBinaryXentLoss(sigmoid=do_sigmoid, softmax=do_softmax,
                                              smooth_num=smooth_num, smooth_den=smooth_den, batch_version=batch_version)
        print(f"[LOSS] Using Custom loss, Dice + Xent with do_sigmoid={do_sigmoid}, do_softmax={do_softmax},"
              f"Dice with {smooth_num} at numerator and {smooth_den} at denominator, "
              f" squared_pred={squared_pred} and batch_version={batch_version}")
    else:
        raise IOError("Unrecognized loss type")

    spacings = spacing[:2]
    sizes = patch_size[:2]
    best_dice, best_epoch = (nr_out_channels - 1) * [0], (nr_out_channels - 1) * [0]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=nr_out_channels,
        kernel_size=kernels,
        strides=strides,
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    ).to(current_device)
    print(net)

    opt = torch.optim.SGD(net.parameters(), lr=float(config_info['training']['lr']), momentum=0.95)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda epoch: (1 - epoch / config_info['training']['nr_train_epochs']) ** 0.9
    )

    """
    MONAI evaluator
    """
    # val_post_transforms = Compose(
    #     [
    #         Activationsd(keys="pred", sigmoid=True),
    #     ]
    # )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=os.path.join(out_model_dir, "valid"),
                                output_transform=lambda x: None,
                                global_epoch_transform=lambda x: trainer.state.iteration),
        CheckpointSaver(save_dir=out_model_dir, save_dict={"net": net, "opt": opt}, save_key_metric=True,
                        file_prefix='best_valid'),
    ]
    if config_info['output']['val_image_to_tensorboad']:
        val_handlers.append(TensorBoardImageHandler(log_dir=os.path.join(out_model_dir, "valid"),
                                                    batch_transform=lambda x: (x["image"], x["label"]),
                                                    output_transform=lambda x: x["pred"], interval=2))

    # Define customized evaluator
    class DynUNetEvaluator(SupervisedEvaluator):
        def _iteration(self, engine, batchdata):
            inputs, targets = self.prepare_batch(batchdata)
            inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)
            flip_inputs = torch.flip(inputs, dims=(2, 3))

            def _compute_pred():
                pred = self.inferer(inputs, self.network)
                flip_pred = torch.flip(self.inferer(flip_inputs, self.network), dims=(2, 3))
                return (pred + flip_pred) / 2

            # execute forward computation
            self.network.eval()
            with torch.no_grad():
                if self.amp:
                    with torch.cuda.amp.autocast():
                        predictions = _compute_pred()
                else:
                    predictions = _compute_pred()
            return {"image": inputs, "label": targets, "pred": predictions}

    evaluator = DynUNetEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer2D(roi_size=patch_size, sw_batch_size=4, overlap=0.0),
        post_transform=None,  # NOTE: IN dynUNet this was set to None!
        key_val_metric={
            "Mean_dice": MeanDice(
                include_background=False,
                to_onehot_y=True,
                mutually_exclusive=True,
                output_transform=lambda x: (x["pred"], x["label"]),
            )
        },
        val_handlers=val_handlers,
        amp=False,
    )

    """
    MONAI trainer
    """
    # train_post_transforms = Compose(
    #     [
    #         Activationsd(keys="pred", sigmoid=True),
    #     ]
    # )

    validation_every_n_epochs = config_info['training']['validation_every_n_epochs']
    epoch_len = len(train_ds) // train_loader.batch_size
    validation_every_n_iters = validation_every_n_epochs * epoch_len

    writer_train = SummaryWriter(log_dir=os.path.join(out_model_dir, "train"))
    train_handlers = [
        LrScheduleHandler(lr_scheduler=scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=validation_every_n_iters, epoch_level=False),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(summary_writer=writer_train,
                                log_dir=os.path.join(out_model_dir, "train"), tag_name="Loss",
                                output_transform=lambda x: x["loss"],
                                global_epoch_transform=lambda x: trainer.state.iteration),
        CheckpointSaver(save_dir=out_model_dir, save_dict={"net": net, "opt": opt},
                        save_final=True,
                        # save_key_metric=True, key_metric_name='Mean_dice', key_metric_n_saved=1,
                        save_interval=2, epoch_level=True,
                        n_saved=config_info['output']['max_nr_models_saved']),
    ]
    if model_to_load is not None:
        train_handlers.append(CheckpointLoader(load_path=model_to_load, load_dict={"net": net, "opt": opt}))

    # define customized trainer
    class DynUNetTrainer(SupervisedTrainer):
        def _iteration(self, engine, batchdata):
            inputs, targets = self.prepare_batch(batchdata)
            inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)

            def _compute_loss(preds, label):
                labels = [label] + [interpolate(label, pred.shape[2:]) for pred in preds[1:]]
                return sum([0.5 ** i * self.loss_function(p, l) for i, (p, l) in enumerate(zip(preds, labels))])

            self.network.train()
            self.optimizer.zero_grad()
            if self.amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.inferer(inputs, self.network)
                    loss = _compute_loss(predictions, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.inferer(inputs, self.network)
                loss = _compute_loss(predictions, targets).mean()
                loss.backward()
                self.optimizer.step()
            return {"image": inputs, "label": targets, "pred": predictions, "loss": loss.item()}

    trainer = DynUNetTrainer(
        device=current_device,
        max_epochs=config_info['training']['nr_train_epochs'],
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        post_transform=None,
        key_train_metric=None,
        # key_train_metric={
        #     "Mean_dice": MeanDice(
        #         include_background=False,
        #         to_onehot_y=True,
        #         mutually_exclusive=True,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        amp=False,
    )

    """
    Run training
    """
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()