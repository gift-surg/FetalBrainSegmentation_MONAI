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
from torchsummary import summary
from torch.nn.modules.loss import BCEWithLogitsLoss, BCELoss

# sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
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
from monai.utils.misc import is_scalar
import warnings
import logging

from io_utils import create_data_list
from sliding_window_inference import sliding_window_inference
from custom_ignite_engines import create_supervised_trainer_with_clipping, create_evaluator_with_sliding_window
from custom_unet import CustomUNet
from custom_losses import DiceAndBinaryXentLoss, DiceLoss_noSmooth, TverskyLoss_noSmooth
from custom_metrics import MeanDiceAndBinaryXentMetric, BinaryXentMetric, TverskyMetric
from custom_transform import ConverToOneHotd

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


class MyTensorBoardImageHandler(object):
    """TensorBoardImageHandler is an ignite Event handler that can visualise images, labels and outputs as 2D/3D images.
    2D output (shape in Batch, channel, H, W) will be shown as simple image using the first element in the batch,
    for 3D to ND output (shape in Batch, channel, H, W, D) input, each of ``self.max_channels`` number of images'
    last three dimensions will be shown as animated GIF along the last axis (typically Depth).

    It's can be used for any Ignite Engine (trainer, validator and evaluator).
    User can easily added it to engine for any expected Event, for example: ``EPOCH_COMPLETED``,
    ``ITERATION_COMPLETED``. The expected data source is ignite's ``engine.state.batch`` and ``engine.state.output``.

    Default behavior:
        - Show y_pred as images (GIF for 3D) on TensorBoard when Event triggered,
        - need to use ``batch_transform`` and ``output_transform`` to specify
          how many images to show and show which channel.
        - Expects ``batch_transform(engine.state.batch)`` to return data
          format: (image[N, channel, ...], label[N, channel, ...]).
        - Expects ``output_transform(engine.state.output)`` to return a torch
          tensor in format (y_pred[N, channel, ...], loss).

     """

    def __init__(self,
                 summary_writer=None,
                 batch_transform=lambda x: x,
                 output_transform=lambda x: x,
                 global_iter_transform=lambda x: x,
                 index=None,
                 max_channels=1,
                 max_frames=64):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            batch_transform (Callable): a callable that is used to transform the
                ``ignite.engine.batch`` into expected format to extract several label data.
            output_transform (Callable): a callable that is used to transform the
                ``ignite.engine.output`` into expected format to extract several output data.
            global_iter_transform (Callable): a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
            index (int): plot which element in a data batch, default is the first element.
            max_channels (int): number of channels to plot.
            max_frames (int): number of frames for 2D-t plot.
        """
        self._writer = SummaryWriter() if summary_writer is None else summary_writer
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_frames = max_frames
        self.max_channels = max_channels

    def __call__(self, engine):
        step = self.global_iter_transform(engine.state.iteration)

        show_images = self.batch_transform(engine.state.batch)[0]
        if torch.is_tensor(show_images):
            show_images = show_images.detach().cpu().numpy()

        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise ValueError('output_transform(engine.state.output)[0] must be an ndarray or tensor.')
            if self.index is None:
                self.index = [0]
            elif self.index in ['all', 'ALL', 'All']:
                print(show_images.shape)
                self.index = range(0, show_images.shape[0])
                print(self.index)
            for idx in self.index:
                plot_2d_or_3d_image(show_images, step, self._writer, idx,
                                    self.max_channels, self.max_frames, 'input_0_' + str(idx))

        show_labels = self.batch_transform(engine.state.batch)[1]
        if torch.is_tensor(show_labels):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise ValueError('batch_transform(engine.state.batch)[1] must be an ndarray or tensor.')
            for idx in self.index:
                plot_2d_or_3d_image(show_labels, step, self._writer, idx,
                                    self.max_channels, self.max_frames, 'input_1_' + str(idx))

        show_outputs = self.output_transform(engine.state.output)
        if torch.is_tensor(show_outputs):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise ValueError('output_transform(engine.state.output) must be an ndarray or tensor.')
            for idx in self.index:
                plot_2d_or_3d_image(show_outputs, step, self._writer, idx,
                                    self.max_channels, self.max_frames, 'output_' + str(idx))

        self._writer.flush()


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

    torch.cuda.set_device(cuda_device)
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
    train_transforms = Compose([
        LoadNiftid(keys=['img', 'seg']),
        ConverToOneHotd(keys=['seg'], labels=seg_labels),
        AddChanneld(keys=['img']),
        NormalizeIntensityd(keys=['img']),
        Resized(keys=['img', 'seg'], spatial_size=[96, 96, -1], mode=["trilinear", "nearest"]),
        RandSpatialCropd(keys=['img', 'seg'], roi_size=[96, 96, 1], random_size=False),
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
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size_train,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    check_train_data = monai.utils.misc.first(train_loader)
    print("Training data tensor shapes")
    print(check_train_data['img'].shape, check_train_data['seg'].shape)

    # data preprocessing for validation:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - apply whitening
    # - resize to (96, 96) in-plane (preserve z-direction)
    if sliding_window_validation:
        val_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            ConverToOneHotd(keys=['seg'], labels=seg_labels),
            AddChanneld(keys=['img']),
            NormalizeIntensityd(keys=['img']),
            Resized(keys=['img', 'seg'], spatial_size=[96, 96, -1], mode=["trilinear", "nearest"]),
            ToTensord(keys=['img', 'seg'])
        ])
        do_shuffle = False
        collate_fn_to_use = None
    else:
        # - add extraction of 2D slices from validation set to emulate how loss is computed at training
        val_transforms = Compose([
            LoadNiftid(keys=['img', 'seg']),
            ConverToOneHotd(keys=['seg'], labels=seg_labels),
            AddChanneld(keys=['img']),
            NormalizeIntensityd(keys=['img']),
            Resized(keys=['img', 'seg'], spatial_size=[96, 96], mode=["trilinear", "nearest"]),
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
    check_valid_data = monai.utils.misc.first(val_loader)
    print("Validation data tensor shapes")
    print(check_valid_data['img'].shape, check_valid_data['seg'].shape)

    """
    Network preparation
    """
    # # Create UNet, DiceLoss and Adam optimizer.
    net = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=nr_out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    # net = CustomUNet()
    print("Model summary:")
    summary(net, input_data=(1, 96, 96))

    smooth = None
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = False
        do_softmax = True
    if loss_type == "Dice":
        loss_function = monai.losses.DiceLoss(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        smooth = 1e-5
        print(f"[LOSS] Using monai.losses.DiceLoss with smooth = {smooth}, do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Xent":
        loss_function = BCEWithLogitsLoss(reduction="mean")
        print("[LOSS] Using BCEWithLogitsLoss")
    elif loss_type == "Dice_nosmooth":
        loss_function = DiceLoss_noSmooth(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Dice with no smooth at numerator, do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Tversky":
        loss_function = monai.losses.TverskyLoss(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using monai.losses.TverskyLoss with do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Tversky_nosmooth":
        loss_function = TverskyLoss_noSmooth(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Tversky with no smooth at numerator with do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
    elif loss_type == "Dice_Xent":
        loss_function = DiceAndBinaryXentLoss(do_sigmoid=do_sigmoid, do_softmax=do_softmax)
        print(f"[LOSS] Using Custom loss, Dice + Xent with do_sigmoid={do_sigmoid}, do_softmax={do_softmax}")
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

    current_device = torch.cuda.current_device()
    if lr_decay is not None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=lr_decay, last_epoch=-1)

    """
    Set ignite trainer
    """
    # function to manage batch at training
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch['img'].to(device), batch['seg'].to(device)), device, non_blocking)

    trainer = create_supervised_trainer_with_clipping(model=net, optimizer=opt, loss_fn=loss_function,
                                                      device=current_device, non_blocking=False, prepare_batch=prepare_batch,
                                                      clip_norm=clipping, smooth_loss=smooth)
    print("Using gradient norm clipping at max = {}".format(clipping))

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
    train_stats_handler = StatsHandler(iteration_print_logger=my_iteration_print_logger,
                                       name='trainer'
                                       )
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    writer_train = SummaryWriter(log_dir=os.path.join(out_model_dir, "train"))
    train_tensorboard_stats_handler = TensorBoardStatsHandler(summary_writer=writer_train)
    train_tensorboard_stats_handler.attach(trainer)

    # train_tensorboard_image_handler = MyTensorBoardImageHandler(
    #     summary_writer=writer_train,
    #     batch_transform=lambda batch: (batch['img'], batch['seg']),
    #     output_transform=lambda output: None,
    #     global_iter_transform=lambda x: trainer.state.iteration,
    #     index=range(0, 6)
    # )
    # trainer.add_event_handler(
    #     event_name=Events.ITERATION_COMPLETED(every=17088), handler=train_tensorboard_image_handler)

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
        # "Loss": BinaryXentMetric(add_sigmoid=True, to_onehot_y=False),
        # "Loss": MeanDiceAndBinaryXentMetric(add_sigmoid=True, to_onehot_y=False),
        "Loss": 1.0 - MeanDice(sigmoid=True, to_onehot_y=False),
        # "Loss": TverskyMetric(add_sigmoid=True, to_onehot_y=False),
        "Mean_Dice": MeanDice(sigmoid=True, to_onehot_y=False)
    }

    if sliding_window_validation:

        # function to manage validation with sliding window inference
        def prepare_sliding_window_inference(x, model):
            return sliding_window_inference(inputs=x, roi_size=(96, 96, 1), sw_batch_size=batch_size_valid,
                                            predictor=model)
        print("3D evaluator is used")
        evaluator = create_evaluator_with_sliding_window(model=net, metrics=val_metrics, device=current_device,
                                                         non_blocking=True, prepare_batch=prepare_batch,
                                                         sliding_window_inference=prepare_sliding_window_inference)
    else:
        # ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
        # user can add output_transform to return other values
        print("2D evaluator is used")
        evaluator = create_supervised_evaluator(model=net, metrics=val_metrics, device=current_device,
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
    state = trainer.run(train_loader, nr_train_epochs)
    print("Done!")


if __name__ == "__main__":
    main()
