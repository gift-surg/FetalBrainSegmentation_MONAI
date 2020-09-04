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

import numpy as np
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.transforms.compose import MapTransform

from monai.config import IndexSelection, KeysCollection
from monai.utils import NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.transforms import DivisiblePad, SpatialCrop
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


class ConverToOneHotd(MapTransform):
    """
    Convert multi-class label to One Hot Encoding:
    """

    def __init__(self, keys, labels):
        """
        Args:

        """
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            for n in self.labels:
                result.append(d[key] == n)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class MinimumPadd(MapTransform):
    """
    Pad the input data, so that the spatial sizes are at least of size `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    def __init__(
        self, keys: KeysCollection, k: Union[Sequence[int], int], mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
        See also :py:class:`monai.transforms.SpatialPad`
        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.k = k
        self.padder = DivisiblePad(k=k)

    def __call__(self, data):
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            spatial_shape = np.array(d[key].shape[1:])
            k = np.array(fall_back_tuple(self.k, (1,) * len(spatial_shape)))
            if np.any(spatial_shape < k):
                d[key] = self.padder(d[key], mode=m)
        return d


def generate_spatial_bounding_box_anisotropic_margin(
        img: np.ndarray,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: Optional[Sequence[int]] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin to all dims of the bounding box.
    """
    data = img[[*(ensure_tuple(channel_indices))]] if channel_indices is not None else img
    data = np.any(select_fn(data), axis=0)
    nonzero_idx = np.nonzero(data)

    if isinstance(margin, int):
        margin = ensure_tuple_rep(margin, len(data.shape))
    margin = [m if m > 0 else 0 for m in margin]
    assert len(data.shape) == len(margin), "defined margin has different number of dimensions than input"

    box_start = list()
    box_end = list()
    for i in range(data.ndim):
        assert len(nonzero_idx[i]) > 0, f"did not find nonzero index at spatial dim {i}"
        box_start.append(max(0, np.min(nonzero_idx[i]) - margin[i]))
        box_end.append(min(data.shape[i], np.max(nonzero_idx[i]) + margin[i] + 1))
    return box_start, box_end


class CropForegroundAnisotropicMargind(MapTransform):
    """
    Expanded version of :py:class:`monai.transforms.CropForegroundd`, which allows to define different margin values
    per dimension.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """

    def __init__(
            self,
            keys: KeysCollection,
            source_key: str,
            select_fn: Callable = lambda x: x > 0,
            channel_indices: Optional[IndexSelection] = None,
            margin: Optional[Sequence[int]] = 0,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin to dims of the bounding box.
        """
        super().__init__(keys)
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box_anisotropic_margin(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin
        )
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        for key in self.keys:
            d[key] = cropper(d[key])
        return d