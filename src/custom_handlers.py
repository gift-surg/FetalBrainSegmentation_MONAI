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
from typing import Callable, Optional

import numpy as np
import torch

from monai.utils import exact_version, optional_import, is_scalar
from monai.visualize import plot_2d_or_3d_image

Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")
Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

DEFAULT_TAG = "Loss"


class MyTensorBoardImageHandler(object):
    """
    Slight modification of TensorBoardImageHandler to allow for displaying multiple images within the batch.
    TensorBoardImageHandler is an Ignite Event handler that can visualise images, labels and outputs as 2D/3D images.
    2D output (shape in Batch, channel, H, W) will be shown as simple image using the first element in the batch,
    for 3D to ND output (shape in Batch, channel, H, W, D) input, each of ``self.max_channels`` number of images'
    last three dimensions will be shown as animated GIF along the last axis (typically Depth).

    It can be used for any Ignite Engine (trainer, validator and evaluator).
    User can easily add it to engine for any expected Event, for example: ``EPOCH_COMPLETED``,
    ``ITERATION_COMPLETED``. The expected data source is ignite's ``engine.state.batch`` and ``engine.state.output``.

    Default behavior:
        - Show y_pred as images (GIF for 3D) on TensorBoard when Event triggered,
        - Need to use ``batch_transform`` and ``output_transform`` to specify
          how many images to show and show which channel.
        - Expects ``batch_transform(engine.state.batch)`` to return data
          format: (image[N, channel, ...], label[N, channel, ...]).
        - Expects ``output_transform(engine.state.output)`` to return a torch
          tensor in format (y_pred[N, channel, ...], loss).

     """

    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        max_frames: int = 64,
    ):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            log_dir: if using default SummaryWriter, write logs to this directory, default is `./runs`.
            interval: plot content from engine.state every N epochs or every N iterations, default is 1.
            epoch_level: plot content from engine.state every N epochs or N iterations. `True` is epoch level,
                `False` is iteration level.
            batch_transform: a callable that is used to transform the
                ``ignite.engine.batch`` into expected format to extract several label data.
            output_transform: a callable that is used to transform the
                ``ignite.engine.output`` into expected format to extract several output data.
            global_iter_transform: a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
            index: plot which element in a data batch, default is the first element.
            max_channels: number of channels to plot.
            max_frames: number of frames for 2D-t plot.
        """
        self._writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_frames = max_frames
        self.max_channels = max_channels

    def attach(self, engine) -> None:
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

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
