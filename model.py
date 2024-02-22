import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import MobileNet_V3_Small_Weights as mweights

import numpy as np

### encoder definition ###


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
encoder = torchvision.models.mobilenet_v3_small(weights=mweights.DEFAULT)
encoder.classifier = Identity()
encoder.avgpool = Identity()


### decoder definition ###


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

decoder = Decoder()
### LSTM model definition ###

class EncoderDecoderLSTM(nn.Module):
    def __init__(self):
        super(EncoderDecoderLSTM, self).__init__()
        self.rnn = nn.LSTM() #nn.LSTM(input_size, hidden_size, num_layers, dropout=p), where input_size = encoder_context_vector_shape
        self.encoder = encoder
        self.decoder = decoder