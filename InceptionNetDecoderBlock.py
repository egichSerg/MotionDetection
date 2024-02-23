import torch
import torch.nn as nn
import torchvision

import numpy as np

from typing import Any

from torchvision.models import MobileNet_V3_Small_Weights as mweights
from torchvision.models import mobilenet_v3_small as mnet


### for additional classes


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        