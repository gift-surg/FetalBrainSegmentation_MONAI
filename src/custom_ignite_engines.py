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

from typing import Callable, Optional

import torch
from monai.inferers import SimpleInferer
from monai.utils import exact_version, optional_import

from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import default_prepare_batch
from monai.engines.evaluator import Evaluator

Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")

from ignite.engine.engine import Engine
from ignite.engine import _prepare_batch

from sliding_window_inference import sliding_window_inference


def create_supervised_trainer_with_clipping(model, optimizer, loss_fn,
                                            device=None, non_blocking=False,
                                            prepare_batch=_prepare_batch,
                                            output_transform=lambda x, y, y_pred, loss: loss.item(),
                                            clip_norm=None, smooth_loss=None):
    """
    Factory function for creating a trainer for supervised models with possibility to define gradient clipping.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`
        clip_norm (float, optional): value to use to clip the norm of the gradients. If None, no clipping is applied

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        if smooth_loss is not None:
            loss = loss_fn(y_pred, y, smooth=smooth_loss)
        else:
            loss = loss_fn(y_pred, y)
        loss.backward()
        if clip_norm is not None:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            except:
                print("Gradient clipping failed.")
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def _sliding_window_inference(x, model):
    """
    Prepare inputs for sliding window inference
    :param x: input data
    :param model: predictor
    :return: inference for input x using the provided model and a sliding window approach
    """
    return sliding_window_inference(inputs=x, roi_size=(96, 96, 1), sw_batch_size=1, predictor=model)


def create_evaluator_with_sliding_window(model, metrics=None,
                                         device=None, non_blocking=False,
                                         prepare_batch=_prepare_batch,
                                         sliding_window_inference=_sliding_window_inference,
                                         output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models, using a sliding window inference approach.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        sliding_window_inference (callable, optional): function that receives the input x and the model and returns the
            inference result for x using a sliding window approach defined by the function.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = sliding_window_inference(x, model)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class SupervisedEvaluatorCropping(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        val_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): use the network to run model forward.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        post_transform (Transform): execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        val_handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader,
        network,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer=SimpleInferer(),
        post_transform=None,
        key_val_metric=None,
        additional_metrics=None,
        val_handlers=None,
    ):
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
        )

        self.network = network
        self.inferer = inferer

    def _iteration(self, engine: Engine, batchdata):
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (TransformContext, ndarray): input data for this iteration.

        Raises:
            ValueError: must provide batch data for current iteration.

        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")
        inputs, targets = self.prepare_batch(batchdata)
        inputs = inputs.to(engine.state.device)
        if targets is not None:
            targets = targets.to(engine.state.device)

        # execute forward computation
        self.network.eval()
        with torch.no_grad():
            predictions = self.inferer(inputs, self.network)

        return {Keys.IMAGE: inputs, 'mask': targets, Keys.PRED: predictions}
