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

import copy
from typing import Callable, Union

import torch
import torch.nn.functional as F

from monai.inferers import Inferer
from monai.inferers.utils import sliding_window_inference
from monai.utils import BlendMode
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple


class SlidingWindowInfererWithResize(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().

    Args:
        roi_size (list, tuple): the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        resize_3d (bool): defines whether to perform resizing in 3D or only in the in-plane direction
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

    Note:
        the "sw_batch_size" here is to run a batch of window slices of 1 input image,
        not batch size of input images.

    """

    def __init__(
        self, roi_size, resize_3d: bool = False, sw_batch_size: int = 1,
            overlap: float = 0.25, mode: Union[BlendMode, str] = BlendMode.CONSTANT
    ):
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.resize_3d = resize_3d
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)

    def __call__(self, inputs: torch.Tensor, network):
        """
        Unified callable function API of Inferers.

        Args:
            inputs (torch.tensor): model input data for inference.
            network (Network): target model to execute inference.

        """

        orig_size = list(inputs.shape)
        resized_size = copy.deepcopy(orig_size)
        resized_size[2] = self.roi_size[0]
        resized_size[3] = self.roi_size[1]
        if self.resize_3d:
            resized_size[4] = self.roi_size[2]
        inputs_resize = torch.nn.functional.interpolate(inputs, size=resized_size[2:], mode='trilinear')
        print(inputs_resize.shape)
        outputs = sliding_window_inference(inputs_resize, self.roi_size, self.sw_batch_size, network,
                                           self.overlap, self.mode)
        outputs = torch.nn.functional.interpolate(outputs, size=orig_size[2:], mode='nearest')

        return outputs
