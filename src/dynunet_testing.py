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

import os
import sys
import yaml
import argparse
import logging
import torch

from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from monai.config import print_config
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.networks.nets import DynUNet
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, MeanDice, SegmentationSaver, StatsHandler
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    ToTensord,
    Activationsd,
)

from io_utils import create_data_list
from custom_inferer import SlidingWindowInferer2DWithResize, SlidingWindowInferer2D
from custom_transform import ConverToOneHotd, InPlaneSpacingd


def main():
    """
    Read input and configuration parameters
    """
    parser = argparse.ArgumentParser(description='Run inference with dynUnet with MONAI.')
    parser.add_argument('--config', dest='config', metavar='config', type=str,
                        help='config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config_info = yaml.load(f, Loader=yaml.FullLoader)

    # print MONAI config information
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # print to log the parameter setups
    print(yaml.dump(config_info))

    # GPU params
    num_workers = config_info['device']['num_workers']
    # inference params
    nr_out_channels = config_info['inference']['nr_out_channels']
    inplane_size = config_info["inference"]["inplane_size"]
    spacing = config_info["inference"]["spacing"]
    batch_size_inference = config_info['inference']['batch_size_inference']
    # temporary check as sliding window inference does not accept higher batch size
    assert batch_size_inference == 1
    prob_thr = config_info['inference']['probability_threshold']
    model_to_load = config_info['inference']['model_to_load']
    if not os.path.exists(model_to_load):
        raise IOError('Trained model not found')
    # data params
    data_root = config_info['data']['data_root']
    inference_list = config_info['data']['inference_list']
    # output saving
    out_dir = config_info['output']['out_dir']
    out_postfix = config_info['output']['out_postfix']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    patch_size = config_info["inference"]["inplane_size"] + [1]
    print(f"Considering patch size = {patch_size}")

    print("\n#### GPU INFORMATION ###")
    print(f"Using device number: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name()}")
    print(f"Device available: {torch.cuda.is_available()}\n")

    """
    Data Preparation
    """
    val_files = create_data_list(data_folder_list=data_root,
                                 subject_list=inference_list,
                                 img_postfix='_Image',
                                 is_inference=True)

    print(len(val_files))
    print(val_files[0])
    print(val_files[-1])

    # data preprocessing for inference:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - apply whitening
    # - NOTE: resizing needs to be applied afterwards, otherwise it cannot be remapped back to original size
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image"]),
            AddChanneld(keys=["image"]),
            # CropForegroundd(keys=["image"], source_key="image"), # I should add this, need to find post-transform to invert it!
            InPlaneSpacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear",
            ),
            # SpatialPadd(keys=["image"], spatial_size=patch_size, mode=["constant"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size_inference,
                            num_workers=num_workers)

    def prepare_batch(batchdata):
        assert isinstance(batchdata, dict), "prepare_batch expects dictionary input data."
        return (
            (batchdata["image"], batchdata["label"])
            if "label" in batchdata
            else (batchdata["image"], None)
        )

    """
    Network preparation
    """
    # TODO: remove this bit and print it from training!!!!
    spacings = spacing[:2]
    sizes = patch_size[:2]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    current_device = torch.device("cuda:0")
    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=nr_out_channels,
        kernel_size=kernels,
        strides=strides,
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    ).to(current_device)

    """
    Set ignite evaluator to perform inference
    """
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = False
        do_softmax = True
    else:
        raise Exception("incompatible number of output channels")
    print(f"Using sigmoid={do_sigmoid} and softmax={do_softmax} as final activation")
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=do_sigmoid, softmax=do_softmax),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=model_to_load, load_dict={"net": net}),
        SegmentationSaver(
            output_dir=out_dir, output_ext='.nii.gz', output_postfix=out_postfix,
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: output["pred"],
        ),
    ]

    # Define customized evaluator
    class DynUNetEvaluator(SupervisedEvaluator):
        def _iteration(self, engine, batchdata):
            inputs, targets = self.prepare_batch(batchdata)
            inputs = inputs.to(engine.state.device)
            if targets is not None:
                targets = targets.to(engine.state.device)
            flip_inputs = torch.flip(inputs, dims=(2, 3))

            def _compute_pred():
                pred = self.inferer(inputs, self.network)
                flip_pred = torch.flip(self.inferer(flip_inputs, self.network), dims=(2, 3))
                return (pred + flip_pred) / 2

            # execute forward computation
            self.network.eval()
            with torch.no_grad():
                if self.amp:
                    with torch.cuda.amp.autocast():
                        predictions = _compute_pred()
                else:
                    predictions = _compute_pred()
            return {"image": inputs, "label": targets, "pred": predictions}

    evaluator = DynUNetEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        inferer=SlidingWindowInferer2D(roi_size=patch_size, sw_batch_size=4, overlap=0.0),
        post_transform=val_post_transforms,
        # key_val_metric={
        #     "Mean_dice": MeanDice(
        #         include_background=False,
        #         to_onehot_y=True,
        #         mutually_exclusive=True,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        val_handlers=val_handlers,
        amp=False,
    )

    """
    Run inference
    """
    evaluator.run()
    print("Done!")


if __name__ == '__main__':
    main()
