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

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, BCELoss

sys.path.append("/mnt/data/mranzini/Desktop/GIFT-Surg/FBS_Monai/MONAI")
import monai
from monai.networks.utils import one_hot


class DiceAndBinaryXentLoss(_Loss):
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

        self.dice_loss_fn = monai.losses.DiceLoss(do_sigmoid=self.do_sigmoid,
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





