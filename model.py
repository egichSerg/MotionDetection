import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import MobileNet_V3_Small_Weights as mweights
from InceptionNetDecoderBlock import DecoderBlock

import numpy as np

### encoder definition ###


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# shape for encoder: [1, 3, 640, 480]    
encoder = torchvision.models.mobilenet_v3_small(weights=mweights.DEFAULT)
encoder.classifier = Identity()
encoder.avgpool = Identity()


### decoder definition ###


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = DecoderBlock
        
        self.deconv1 = self.block(in_channels=1, out_channels=2) # TODO: define number of channels
        
    def forward(self, x):
        ### deconv 5 times
        ### after each deconv concat with corresponding conv step
        
        return output
    
decoder = Decoder()


### LSTM model definition ###


class EncoderDecoderLSTM(nn.Module):
    def __init__(self):
        super(EncoderDecoderLSTM, self).__init__()
        self.rnn = nn.LSTM() #nn.LSTM(input_size, hidden_size, num_layers, dropout=p), where input_size = encoder_context_vector_shape
        self.encoder = encoder
        self.decoder = decoder