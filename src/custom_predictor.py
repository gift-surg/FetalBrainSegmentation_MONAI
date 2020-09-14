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
import torch
import monai


class Predict2DFrom3D:
    """
    Crop 2D slices from 3D inputs and perform 2D predictions
    """
    def __init__(self,
                 predictor):
        """

        :param predictor:
        """
        self.predictor = predictor

    def __call__(self, data):
        # squeeze dimensions equal to 1
        orig_size = list(data.shape)
        data_size = list(data.shape[2:])
        for idx_dim in range(2, 2+len(data_size)):
            if data_size[idx_dim-2] == 1:
                data = torch.squeeze(data, dim=idx_dim)
        predictions = self.predictor(data)  # batched patch segmentation
        new_size = copy.deepcopy(orig_size)
        new_size[1] = predictions.shape[1]   # keep original data shape, but take channel dimension from the prediction
        predictions = torch.reshape(predictions, new_size)
        return predictions
