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
from monai.engines.workflow import Workflow
from monai.engines.trainer import Trainer

Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")


class SupervisedTrainerClipping(Trainer):
    """
    Standard supervised training method with image and label, inherits from trainer and Workflow.
    Allows for gradient clipping

    Args:
        device (torch.device): an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        train_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): to train with this network.
        optimizer (Optimizer): the optimizer associated to the network.
        loss_function (Loss): the loss function associated to the optimizer.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        amp: whether to enable auto-mixed-precision training, reserved.
        post_transform (Transform): execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        train_handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        grad_clipping (float): if not None, defines the threshold value for clipping the norm of gradients

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader,
        network,
        optimizer,
        loss_function,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer=SimpleInferer(),
        amp: bool = True,
        post_transform=None,
        key_train_metric: Optional[Metric] = None,
        additional_metrics=None,
        train_handlers=None,
        grad_clipping=None
    ) -> None:
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            amp=amp,
            data_loader=train_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            post_transform=post_transform,
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = inferer
        self.grad_clipping = grad_clipping

    def _iteration(self, engine: Engine, batchdata):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (dict or array of tensor): input data for this iteration.

        Raises:
            ValueError: must provide batch data for current iteration.

        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")
        inputs, targets = self.prepare_batch(batchdata)
        inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)

        self.network.train()
        self.optimizer.zero_grad()
        # execute forward computation
        predictions = self.inferer(inputs, self.network)
        # compute loss
        loss = self.loss_function(predictions, targets).mean()
        loss.backward()
        if self.grad_clipping is not None:
            try:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clipping)
            except:
                print("Gradient clipping failed.")
        self.optimizer.step()

        return {Keys.IMAGE: inputs, Keys.LABEL: targets, Keys.PRED: predictions, Keys.LOSS: loss.item()}
