# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .caltech101 import Caltech101Processor
from .cifar import CIFAR10Processor, CIFAR100Processor
from .clevr import CLEVRProcessor
from .dmlab import DmlabProcessor
from .dsprites import DSpritesProcessor
from .dtd import DTDProcessor
from .eurosat import EuroSATProcessor
from .fgvc_aircraft import FGVCAircraftProcessor
from .flowers102 import Flowers102Processor
from .food101 import Food101Processor
from .imagenet import ImageNet1KProcessor
from .imagenet21k import ImageNet21KProcessor
from .kitti import KITTIProcessor
from .oxfordiiitpets import OxfordIIITPetProcessor
from .patch_camelyon import PatchCamelyonProcessor
from .resisc45 import Resisc45Processor

# from .retinopathy import DiabeticRetinopathyProcessor
from .smallnorb import SmallNORBProcessor
from .stanfordcars import StanfordCarsProcessor
from .sun397 import SUN397Processor
from .svhn import SVHNProcessor

__all__ = (
    "Caltech101Processor",
    "CIFAR10Processor",
    "CIFAR100Processor",
    "CLEVRProcessor",
    "DmlabProcessor",
    "DSpritesProcessor",
    "DTDProcessor",
    "EuroSATProcessor",
    "FGVCAircraftProcessor",
    "Flowers102Processor",
    "Food101Processor",
    "ImageNet1KProcessor",
    "ImageNet21KProcessor",
    "KITTIProcessor",
    "OxfordIIITPetProcessor",
    "PatchCamelyonProcessor",
    "Resisc45Processor",
    # "DiabeticRetinopathyProcessor",
    "SmallNORBProcessor",
    "StanfordCarsProcessor",
    "SUN397Processor",
    "SVHNProcessor",
)
