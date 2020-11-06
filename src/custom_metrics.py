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
from typing import Callable, Optional, Sequence, Union

import torch
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, BCELoss
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from monai.metrics import DiceMetric, compute_meandice
from monai.networks import one_hot
from monai.utils import MetricReduction, exact_version, optional_import
from monai.losses import TverskyLoss


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
        Args:
            weight_dice (Float): weight of the Dice component to the total metric. Default is 1.0
            weight_xent (Float): weight of the Cross Entropy component to the total metric. Default is 1.0
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
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

        self._xent_scores = self.xent_fn(y_pred, y)  # this is already computed over the batch
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


class BinaryXentMetric(Metric):
    """Computes BinaryXentMetric metric from full size Tensor and collects average over batch,
    class-channels, iterations.
    """

    def __init__(
        self,
        add_sigmoid=False,
        include_background=True,
        to_onehot_y=False,
        mutually_exclusive=False,
        logit_thresh=0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True. [Deprecated - Not used]
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
                [Deprecated - Not used]
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False. [Deprecated - Not used]
            logit_thresh (Float): the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
                [Deprecated - Not used]
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
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


class TverskyMetric(Metric):
    """Computes Tversky Loss as a metric from full size Tensor and collects average over batch,
    class-channels, iterations.
    """

    def __init__(
        self,
        add_sigmoid=False,
        include_background=True,
        to_onehot_y=False,
        mutually_exclusive=False,
        alpha=0.5,
        beta=0.5,
        reduction="mean",
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            alpha (Float): weight of false positives
            beta  (Float): weight of false negatives
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.add_sigmoid = add_sigmoid
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

        self.tversky_fn = TverskyLoss(do_sigmoid=self.add_sigmoid,
                                      include_background=self.include_background,
                                      to_onehot_y=self.to_onehot_y,
                                      alpha=self.alpha,
                                      beta=self.beta,
                                      reduction=self.reduction)

        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        if not len(output) == 2:
            raise ValueError("Tversky Metric can only support y_pred and y.")
        y_pred, y = output

        scores = self.tversky_fn(y_pred, y)

        # add all items in current batch
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
            raise NotComputableError("Tversky must have at least one example "
                                     "before it can be computed.")
        return self._sum / self._num_examples


class PercentageSliceDice(Metric):
    def __init__(
        self,
        logit_thresh: float = 0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
        dice_thr: float = 0.9,
        percentage_thr: float = 0.8
    ) -> None:
        """

        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y: whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive: if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            sigmoid: whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            other_act: callable function to replace `sigmoid` as activation layer if needed, Defaults to ``None``.
                for example: `other_act = torch.tanh`.
            logit_thresh: the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            dice_thr: threshold of Dice score to consider a single slice validly segmented
            percentage_thr: threshold of percentage of slices that are required

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.dice_thr = dice_thr
        self.percentage_thr = percentage_thr
        self._sum = 0.0
        self._num_examples = 0
        self.logit_thresh = logit_thresh

    def compute_slice_percentage_with_dice(self,
                                           y_pred: torch.Tensor,
                                           y: torch.Tensor,
                                           include_background: bool = True,
                                           to_onehot_y: bool = True,
                                           mutually_exclusive: bool = True,
                                           sigmoid: bool = False,
                                           other_act: Optional[Callable] = None
    ) -> torch.Tensor:
        """Computes Dice score metric from full size Tensor and collects average.

        Args:
            y_pred: input data to compute, typical segmentation model output.
                it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
            y: ground truth to compute mean dice metric, the first dim is batch.
                example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
                alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
            include_background: whether to skip Dice computation on the first channel of
                the predicted output. Defaults to True.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
                a combination of argmax and to_onehot.  Defaults to False.
            sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
            other_act: callable function to replace `sigmoid` as activation layer if needed, Defaults to ``None``.
                for example: `other_act = torch.tanh`.
            logit_thresh: the threshold value used to convert (for example, after sigmoid if `sigmoid=True`)
                `y_pred` into a binary matrix. Defaults to 0.5.

        Raises:
            ValueError: When ``sigmoid=True`` and ``other_act is not None``. Incompatible values.
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When ``sigmoid=True`` and ``mutually_exclusive=True``. Incompatible values.

        Returns:
            Dice scores per batch and per class, (shape [batch_size, n_classes]).

        Note:
            This method provides two options to convert `y_pred` into a binary matrix
                (1) when `mutually_exclusive` is True, it uses a combination of ``argmax`` and ``to_onehot``,
                (2) when `mutually_exclusive` is False, it uses a threshold ``logit_thresh``
                    (optionally with a ``sigmoid`` function before thresholding).

        """
        n_classes = y_pred.shape[1]
        if sigmoid and other_act is not None:
            raise ValueError("Incompatible values: sigmoid=True and other_act is not None.")
        if sigmoid:
            y_pred = y_pred.float().sigmoid()

        if other_act is not None:
            if not callable(other_act):
                raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
            y_pred = other_act(y_pred)

        if n_classes == 1:
            if mutually_exclusive:
                warnings.warn("y_pred has only one class, mutually_exclusive=True ignored.")
            if to_onehot_y:
                warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
            if not include_background:
                warnings.warn("y_pred has only one channel, include_background=False ignored.")
            # make both y and y_pred binary
            y_pred = (y_pred >= self.logit_thresh).float()
            y = (y > 0).float()
        else:  # multi-channel y_pred
            # make both y and y_pred binary
            if mutually_exclusive:
                if sigmoid:
                    raise ValueError("Incompatible values: sigmoid=True and mutually_exclusive=True.")
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
                y_pred = one_hot(y_pred, num_classes=n_classes)
            else:
                y_pred = (y_pred >= self.logit_thresh).float()
            if to_onehot_y:
                y = one_hot(y, num_classes=n_classes)

        if not include_background:
            y = y[:, 1:] if y.shape[1] > 1 else y
            y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

        assert y.shape == y_pred.shape, "Ground truth one-hot has differing shape (%r) from source (%r)" % (
            y.shape,
            y_pred.shape,
        )
        y = y.float()
        y_pred = y_pred.float()

        # loop over all elements in the batch and
        n_batch_el = y.shape[0]
        f = []
        for b in range(n_batch_el):
            y_b = y[b]
            y_pred_b = y_pred[b]
            n_slices = y_b.shape[-1]
            # count the number of slices with dice above the threshold
            good_slices = 0
            for s in range(n_slices):
                # check if there are foreground pixels in the current slice of current batch element
                fg_slice = y_b[1, ..., s]
                if fg_slice.float().sum().item() == 0:
                    channel_to_use = 0
                else:
                    channel_to_use = 1
                current_intersection = torch.sum(y_b[channel_to_use, ..., s] * y_pred_b[channel_to_use, ..., s])
                current_denominator = torch.sum(y_b[channel_to_use, ..., s]) + \
                                      torch.sum(y_pred_b[channel_to_use, ..., s])
                current_dice = (2.0 * current_intersection) / current_denominator

                if current_dice >= self.dice_thr:
                    good_slices += 1

            f.append(good_slices / n_slices)

        f = torch.FloatTensor(f)
        return f  # returns array of percentages shape: [batch, 1]

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. MeanDice metric can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        score = self.compute_slice_percentage_with_dice(y_pred, y)

        if type(score.item()) in (float, int):
            loop_batch = [score.item()]
        else:
            loop_batch = score.item()
        # count all cases with at least self.percentage_thr of slices with dice >= self.dice_thr
        for s in loop_batch:
            if s >= self.percentage_thr:
                self._sum += 1
            self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self._num_examples == 0:
            raise NotComputableError("PercentageSliceDice must have at least one example before it can be computed.")
        return self._sum / self._num_examples