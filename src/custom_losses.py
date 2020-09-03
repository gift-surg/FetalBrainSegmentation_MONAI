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

import warnings
import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, BCELoss

sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.networks.utils import one_hot


class DiceAndBinaryXentLoss(_Loss):
    """
    Compute a loss function that combines Dice Loss and Binary Cross Entropy:
    L = weight_dice * Dice_loss + weight_xent * Xent_loss
    The weight terms can be set by the user. All other inputs are matched to the inputs of the Dice loss in MONAI
    """
    def __init__(
            self,
            weight_dice=1.,
            weight_xent=1.,
            do_sigmoid=False,
            include_background=True,
            to_onehot_y=False,
            do_softmax=False,
            squared_pred=False,
            jaccard=False,
            reduction="mean"
    ):
        """
        Args:
            weight_dice (float): [DICE] weight factor of the Dice loss in the total loss
            weight_xent (float): [XENT] weight factor of the Binary Cross Entropy loss in the total loss
            do_sigmoid (bool): [DICE and XENT] If True, apply a sigmoid function to the prediction.
            include_background (bool): [DICE] If False channel index 0 (background category)
                is excluded from the calculation.
            to_onehot_y (bool): [DICE] whether to convert `y` into the one-hot format. Defaults to False.
            do_softmax (bool): [DICE] If True, apply a softmax function to the prediction.
            squared_pred (bool): [DICE] use squared versions of targets and predictions in the denominator or not.
            jaccard (bool): [DICE] compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.weight_dice = weight_dice
        self.weight_xent = weight_xent
        self.do_sigmoid = do_sigmoid
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.do_softmax = do_softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

        # self.dice_loss_fn = monai.losses.DiceLoss(do_sigmoid=self.do_sigmoid,
        #                                           do_softmax=self.do_softmax,
        #                                           squared_pred=self.squared_pred,
        #                                           jaccard=self.jaccard,
        #                                           reduction=reduction)
        self.dice_loss_fn = DiceLoss_noSmooth(do_sigmoid=self.do_sigmoid,
                                              do_softmax=self.do_softmax,
                                              squared_pred=self.squared_pred,
                                              jaccard=self.jaccard,
                                              reduction=reduction)

        if self.do_sigmoid:
            self.xent_fn = BCEWithLogitsLoss(reduction=reduction)
        else:
            self.xent_fn = BCELoss(reduction=reduction)

    def forward(self, input, target, smooth=1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan in Dice Computation.
        """
        n_pred_ch = input.shape[1]
        if n_pred_ch > 1:
            raise IOError("DiceAndBinaryXentLoss not yet implemented for multi-class problems")
        dice_loss = self.dice_loss_fn(input, target, smooth)
        xent_loss = self.xent_fn(input, target)
        return self.weight_dice * dice_loss + self.weight_xent * xent_loss


class MultiScaleDice(_Loss):
    """
    Compute a loss function that evaluates Dice loss on S scales and returns the average Dice across scales:
    Based on Ebner E., Wang, G., "An automated framework for localization, segmentation and super-resolution
    reconstruction of fetal brain MRI",  NeuroImage (2020)

    """
    def __init__(
            self,
            dimensions=2,
            number_of_scales=4,
            do_sigmoid=False,
            include_background=True,
            to_onehot_y=False,
            do_softmax=False,
            squared_pred=False,
            jaccard=False,
            reduction="mean"
    ):
        """
        Args:
            dimensions (int):
            number_of_scales (int): number of scales to compute the Dice loss over
            do_sigmoid (bool): [DICE and XENT] If True, apply a sigmoid function to the prediction.
            include_background (bool): [DICE] If False channel index 0 (background category)
                is excluded from the calculation.
            to_onehot_y (bool): [DICE] whether to convert `y` into the one-hot format. Defaults to False.
            do_softmax (bool): [DICE] If True, apply a softmax function to the prediction.
            squared_pred (bool): [DICE] use squared versions of targets and predictions in the denominator or not.
            jaccard (bool): [DICE] compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.dimensions = dimensions
        self.number_of_scales = number_of_scales
        self.do_sigmoid = do_sigmoid
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.do_softmax = do_softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

        if self.dimensions == 2:
            self.avg_pool_fn = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.dimensions == 3:
            self.avg_pool_fn = torch.nn.AvgPool3d(kernel_size=2, stride=2)
            # NOTE: I am downsampling along all directions! makes sense or should I keep z fixed?
        else:
            raise IOError("The number of the input dimensions is not currently supported")

        self.dice_loss_fn = DiceLoss_noSmooth(do_sigmoid=self.do_sigmoid,
                                              do_softmax=self.do_softmax,
                                              squared_pred=self.squared_pred,
                                              jaccard=self.jaccard,
                                              reduction=reduction)

    def forward(self, input, target, smooth=1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan in Dice Computation.
        """
        #TODO assert input and target are either 2D or 3D
        dice_loss = self.dice_loss_fn(input, target, smooth)

        final_size = np.asarray(list(input.shape))[2:] / (2 ** self.number_of_scales)
        assert (
                np.all(final_size >= 1)
        ), f"selected number or scales is too big compared to the input size)"

        for s in range(self.number_of_scales-1):
            input = self.avg_pool_fn(input)
            target = self.avg_pool_fn(target)
            dice_loss += self.dice_loss_fn(input, target, smooth)

        return dice_loss / self.number_of_scales


# exactly the same as Monai's Dice, but smooth factor is removed from the numerator
class DiceLoss_noSmooth(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    union component of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(
        self,
        include_background=True,
        to_onehot_y=False,
        do_sigmoid=False,
        do_softmax=False,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
    ):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            squared_pred (bool): use squared versions of targets and predictions in the denominator or not.
            jaccard (bool): compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if do_sigmoid and do_softmax:
            raise ValueError("do_sigmoid=True and do_softmax=True are not compatible.")
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, input, target, smooth=1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan. Added to the denominator only.
        """
        if self.do_sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.do_softmax:
                warnings.warn("single channel prediction, `do_softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            if self.do_softmax:
                input = torch.softmax(input, 1)
            if self.to_onehot_y:
                target = one_hot(target, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator -= intersection

        f = 1.0 - (2.0 * intersection) / (denominator + smooth)     # this is the line I changed
        if self.reduction == "sum":
            return f.sum()  # sum over the batch and channel dims
        if self.reduction == "none":
            return f  # returns [N, n_classes] losses
        if self.reduction == "mean":
            return f.mean()  # the batch and channel average
        raise ValueError(f"reduction={self.reduction} is invalid.")


# exactly the same as Monai's Tversky, but smooth factor is removed from the numerator
class TverskyLoss_noSmooth(_Loss):

    """
    Compute the Tversky loss defined in:

        Sadegh et al. (2017) Tversky loss function for image segmentation
        using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631

    """

    def __init__(
        self,
        include_background=True,
        to_onehot_y=False,
        do_sigmoid=False,
        do_softmax=False,
        alpha=0.5,
        beta=0.5,
        reduction="mean",
    ):

        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            alpha (float): weight of false positives
            beta  (float): weight of false negatives
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.

        """

        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y

        if do_sigmoid and do_softmax:
            raise ValueError("do_sigmoid=True and do_softmax=True are not compatible.")
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target, smooth=1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan. Added to the denominator only
        """
        if self.do_sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.do_softmax:
                warnings.warn("single channel prediction, `do_softmax=True` ignored.")
            if self.to_onehot_y:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            if not self.include_background:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            if self.do_softmax:
                input = torch.softmax(input, 1)
            if self.to_onehot_y:
                target = one_hot(target, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = self.alpha * torch.sum(p0 * g1, reduce_axis)
        fn = self.beta * torch.sum(p1 * g0, reduce_axis)

        numerator = tp  # + smooth                        # this is the line I changed
        denominator = tp + fp + fn + smooth

        score = 1.0 - numerator / denominator

        if self.reduction == "sum":
            return score.sum()  # sum over the batch and channel dims
        if self.reduction == "none":
            return score  # returns [N, n_classes] losses
        if self.reduction == "mean":
            return score.mean()
        raise ValueError(f"reduction={self.reduction} is invalid.")

