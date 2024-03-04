import torch
import torch.nn as nn

import numpy as np

from mobile_like import Mobile_like_encoder as Encoder
from mobile_like import Mobile_like_decoder as Decoder

### encoder definition ###



### decoder definition ###



### LSTM model definition ###


class EncoderDecoderLSTM(nn.Module):
    def __init__(self):
        super(EncoderDecoderLSTM, self).__init__()
        self.rnn = nn.LSTM() #nn.LSTM(input_size, hidden_size, num_layers, dropout=p), where input_size = encoder_context_vector_shape
        self.encoder = encoder
        self.decoder = decoder