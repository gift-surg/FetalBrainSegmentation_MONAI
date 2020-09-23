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

import numpy as np
from typing import Sequence, Union

from monai.transforms.compose import MapTransform

from monai.config import KeysCollection
from monai.utils import NumpyPadMode, ensure_tuple_rep, fall_back_tuple
from monai.transforms.croppad.array import DivisiblePad
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