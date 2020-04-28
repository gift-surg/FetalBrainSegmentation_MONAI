import os
import sys
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import list_data_collate
from monai.transforms import Compose, LoadNiftid, AddChanneld, AsChannelFirstd, NormalizeIntensityd, Resized, \
     RandSpatialCropd, RandRotated, RandFlipd, Orientationd, ToTensord
from monai.transforms.compose import Transform, MapTransform
from monai.metrics import compute_meandice
from monai.visualize import plot_2d_or_3d_image

from io_utils import create_data_list
from sliding_window_inference import sliding_window_inference


class SqueezeDim(Transform):
    """
    Squeeze undesired unitary dimensions
    """

    def __init__(self, dim=None):
        """
        Args:
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        if dim is not None:
            assert isinstance(dim, int) and dim >= -1, 'invalid channel dimension.'
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (ndarray): numpy arrays with required dimension `dim` removed
        """
        return np.squeeze(img, self.dim)


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:SqueezeDim`.
    """

    def __init__(self, keys, dim=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        super().__init__(keys)
        self.converter = SqueezeDim(dim=dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


def main():

    ############################# parameters that will need to be set with config file
    # GPU params
    cuda_device = 1
    num_workers = 1
    # training and validation params
    loss_type = "Dice"
    batch_size_train = 10
    batch_size_valid = 1
    lr = 1e-3
    nr_train_epochs = 600
    validation_every_n_epochs = 2
    # data params
    data_root = ["/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset/GroupA",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset/GroupB1",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset/GroupB2",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset_extension/GroupC",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset_extension/GroupD",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset_extension/GroupE",
                 "/mnt/data/mranzini/Desktop/GIFT-Surg/Data/NeuroImage_dataset_extension/GroupF"]
    # list of subject IDs to search for data
    list_root = "/mnt/data/mranzini/Desktop/GIFT-Surg/Retraining_with_expanded_dataset/config/file_names"
    training_list = os.path.join(list_root, "list_train_files.txt")
    validation_list = [os.path.join(list_root, "list_validation_h_files.txt"),
                       os.path.join(list_root, "list_validation_p_files.txt")]
    # model saving
    out_model_dir = os.path.join("/mnt/data/mranzini/Desktop/GIFT-Surg/UNet_Monai/runs",
                                 datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    max_nr_models_saved = 5
    #################################

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    torch.cuda.set_device(cuda_device)

    """
    Data Preparation
    """
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
        Resized(keys=['img'], spatial_size=[96, 96], order=1),
        Resized(keys=['seg'], spatial_size=[96, 96], order=0, anti_aliasing=False),
        RandSpatialCropd(keys=['img', 'seg'], roi_size=[96, 96, 1], random_size=False),
        RandRotated(keys=['img', 'seg'], degrees=90, prob=0.2, spatial_axes=[0, 1], order=0, reshape=False),
        RandFlipd(keys=['img', 'seg'], spatial_axis=[0, 1]),
        SqueezeDimd(keys=['img', 'seg'], dim=-1),
        ToTensord(keys=['img', 'seg'])
    ])
    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
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
    val_transforms = Compose([
        LoadNiftid(keys=['img', 'seg']),
        AddChanneld(keys=['img', 'seg']),
        NormalizeIntensityd(keys=['img']),
        Resized(keys=['img'], spatial_size=[96, 96], order=1),
        Resized(keys=['seg'], spatial_size=[96, 96], order=0, anti_aliasing=False),
        ToTensord(keys=['img', 'seg'])
    ])
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size_valid,
                            num_workers=num_workers)
    check_valid_data = monai.utils.misc.first(val_loader)
    print("Validation data tensor shapes")
    print(check_valid_data['img'].shape, check_valid_data['seg'].shape)

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

    """
    Training loop
    """
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=out_model_dir)
    net.to(device)
    for epoch in range(nr_train_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, nr_train_epochs))
        net.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print("%d/%d, train_loss:%0.4f" % (step, epoch_len, loss.item()))
            writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print("epoch %d average loss:%0.4f" % (epoch + 1, epoch_loss))

        if (epoch + 1) % validation_every_n_epochs == 0:
            net.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
                    roi_size = (96, 96, 1)
                    val_outputs = sliding_window_inference(val_images, roi_size, batch_size_valid, net)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=True,
                                             to_onehot_y=False, add_sigmoid=True)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), os.path.join(out_model_dir, 'best_metric_model.pth'))
                    print('saved new best metric model')
                print("current epoch %d current mean dice: %0.4f best mean dice: %0.4f at epoch %d"
                      % (epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_mean_dice', metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag='image')
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag='label')
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag='output')

    print('train completed, best_metric: %0.4f  at epoch: %d' % (best_metric, best_metric_epoch))
    writer.close()


if __name__ == "__main__":
    main()
