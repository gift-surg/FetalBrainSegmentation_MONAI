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

import warnings
import sys
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, BCELoss, CrossEntropyLoss

# sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.networks.utils import one_hot
from monai.utils import LossReduction, Weight


class DiceLossExtended(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    With respect to monai.losses.DiceLoss, this implementation allows for:
    - the use of a "Batch Dice" (batch version) as in the nnUNet implementation. The Dice is computed for the whole
        batch (1 value per class channel), as opposed to being computed for each element in the batch and then averaged
        across the batch.
    - the selection of different smooth terms at numerator and denominator.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        batch_version: bool = False,
        smooth_num: float = 1e-5,
        smooth_den: float = 1e-5
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch_version: if True, a single Dice value is computed for the whole batch per class. If False, the Dice
                is computed per element in the batch and then reduced (sum/average/None) across the batch.
            smooth_num: a small constant to be added to the numerator of Dice to avoid nan.
            smooth_den: a small constant to be added to the denominator of Dice to avoid nan.
        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.batch_version = batch_version
        self.smooth_num = smooth_num
        self.smooth_den = smooth_den

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD]
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        if self.batch_version:
            # reducing only spatial dimensions and batch (not channels)
            reduce_axis = [0] + list(range(2, len(input.shape)))
        else:
            # reducing only spatial dimensions (not batch nor channels)
            reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_num) / (denominator + self.smooth_den)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses or [n_classes] if batch version
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


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
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            batch_version: bool = False,
            smooth_num: float = 1e-5,
            smooth_den: float = 1e-5
    ) -> None:
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
            batch_version: if True, a single Dice value is computed for the whole batch per class. If False, the Dice
                is computed per element in the batch and then reduced (sum/average/None) across the batch.
            smooth_num: a small constant to be added to the numerator of Dice to avoid nan.
            smooth_den: a small constant to be added to the denominator of Dice to avoid nan.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.weight_dice = weight_dice
        self.weight_xent = weight_xent
        self.sigmoid = sigmoid

        # self.dice_loss_fn = monai.losses.DiceLoss(do_sigmoid=self.do_sigmoid,
        #                                           do_softmax=self.do_softmax,
        #                                           squared_pred=self.squared_pred,
        #                                           jaccard=self.jaccard,
        #                                           reduction=reduction)
        self.dice_loss_fn = DiceLossExtended(include_background=include_background,
                                             to_onehot_y=to_onehot_y,
                                             sigmoid=sigmoid,
                                             softmax=softmax,
                                             other_act=other_act,
                                             squared_pred=squared_pred,
                                             jaccard=jaccard,
                                             reduction=reduction,
                                             batch_version=batch_version,
                                             smooth_num=smooth_num,
                                             smooth_den=smooth_den)
        # TODO: if i have one-hot encoding, should I use binary of normal cross entropy?
        # if self.num_classes == 1:
        if self.sigmoid:
            "Using BCEWithLogitsLoss"
            self.xent_fn = BCEWithLogitsLoss(reduction=LossReduction(reduction).value)
        else:
            "Using BCELoss"
            self.xent_fn = BCELoss(reduction=LossReduction(reduction).value)

    def forward(self, input, target, smooth_num=1e-5, smooth_den=1e-5):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth_num: a small constant to be added to the numerator of Dice to avoid nan.
            smooth_den: a small constant to be added to the denominator of Dice to avoid nan.
        """
        # n_pred_ch = input.shape[1]
        # if n_pred_ch > 1:
        #     raise IOError("DiceAndBinaryXentLoss not yet implemented for multi-class problems")
        dice_loss = self.dice_loss_fn(input, target)
        xent_loss = self.xent_fn(input, target)
        return self.weight_dice * dice_loss + self.weight_xent * xent_loss


# TODO: Do the same as for the Dice loss - single version with option to choose smooth_num and smooth_den
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

