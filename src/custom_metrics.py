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

from typing import Callable, Optional, Sequence, Union

import torch
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, BCELoss
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from monai.metrics import compute_meandice


class MeanDiceAndBinaryXentMetric(Metric):
    """Computes MeanDiceAndBinaryXentMetric metric from full size Tensor and collects average over batch,
    class-channels, iterations.
    """

    def __init__(
        self,
        weight_dice=1.,
        weight_xent=1.,
        add_sigmoid=False,
        include_background=True,
        to_onehot_y=False,
        mutually_exclusive=False,
        logit_thresh=0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        #TODO: UPDATE
        Args:
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            logit_thresh (Float): the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.weight_dice = weight_dice
        self.weight_xent = weight_xent
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.add_sigmoid = add_sigmoid
        self.logit_thresh = logit_thresh

        if self.add_sigmoid:
            self.xent_fn = BCEWithLogitsLoss(reduction="mean")
        else:
            self.xent_fn = BCELoss(reduction="mean")
        self._xent_scores = 0

        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0
        self._xent_scores = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        if not len(output) == 2:
            raise ValueError("MeanDiceAndBinaryXentMetric can only support y_pred and y.")
        y_pred, y = output
        scores_dice = compute_meandice(
            y_pred,
            y,
            self.include_background,
            self.to_onehot_y,
            self.mutually_exclusive,
            self.add_sigmoid,
            self.logit_thresh,
        )

        # scores_xent = self.xent_fn(y_pred, y)

        # scores = self.weight_dice * (1. - scores_dice) + self.weight_xent * scores_xent

        self._xent_scores = self.xent_fn(y_pred, y)
        scores = (1. - scores_dice)

        # add all items in current batch (only for Dice)
        for batch in scores:
            not_nan = ~torch.isnan(batch)
            if not_nan.sum() == 0:
                continue
            class_avg = batch[not_nan].mean().item()
            self._sum += class_avg
            self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("MeanDiceAndBinaryXentMetric must have at least one example "
                                     "before it can be computed.")
        return self.weight_dice * (self._sum / self._num_examples) + self.weight_xent * self._xent_scores
        # return self._sum / self._num_examples


class BinaryXentMetric(Metric):
    """Computes BinaryXentMetric metric from full size Tensor and collects average over batch,
    class-channels, iterations.
    """

    def __init__(
        self,
        weight_dice=1.,
        weight_xent=1.,
        add_sigmoid=False,
        include_background=True,
        to_onehot_y=False,
        mutually_exclusive=False,
        logit_thresh=0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        #TODO: UPDATE
        Args:
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            logit_thresh (Float): the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.weight_dice = weight_dice
        self.weight_xent = weight_xent
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.add_sigmoid = add_sigmoid
        self.logit_thresh = logit_thresh

        if self.add_sigmoid:
            self.xent_fn = BCEWithLogitsLoss(reduction="mean")
        else:
            self.xent_fn = BCELoss(reduction="mean")

        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        if not len(output) == 2:
            raise ValueError("BinaryXentMetric can only support y_pred and y.")
        y_pred, y = output

        scores = self.xent_fn(y_pred, y)

        # add all items in current batch
        #TODO: ugly fix!!! scores is already the mean value over the batch, so it's just a single number, need to code it better
        for batch in [scores]:
            not_nan = ~torch.isnan(batch)
            if not_nan.sum() == 0:
                continue
            class_avg = batch[not_nan].mean().item()
            self._sum += class_avg
            self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BinaryXentMetric must have at least one example "
                                     "before it can be computed.")
        return self._sum / self._num_examples