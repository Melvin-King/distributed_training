import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

from transformers import AutoConfig, GPT2Model, GPT2PreTrainedModel

from .pipe import Pipe
from .partition import WithDevice, _retrieve_device
from .model import GPT2ModelCustom, GPT2LMHeadModelCustom

class ExtractFirstItem(nn.Module):
    def __init__(self):
        super(ExtractFirstItem, self).__init__()
    
    def forward(self, x):
        return x[0]


class GPT2BlockWrapper(nn.Module):
    '''
    Class GPT2BlockWrapper may be useful to avoid some type errors.
    '''
    def __init__(self, gpt2_block):
        super().__init__()
        self.gpt2_block = gpt2_block

    def forward(self, hidden_states, *args, **kwargs):
        # Call the original GPT2Block
        output = self.gpt2_block(hidden_states, *args, **kwargs)
        if isinstance(output, tuple):
            # Extract the tensor needed for the next stage
            hidden_states = output[0]
        return hidden_states


class GPT2ModelParallel(GPT2ModelCustom):
    def __init__(self, config):
        super().__init__(config)

    # Homework 2
    def _prepare_pipeline_parallel(self, split_size=1):
        '''
        Prepare the model for pipeline parallelism.

        Hint:
        1. You are suggested to read this .py file before programming
        2. Enable self.pipeline_parallel
        3. Construct an nn.Sequential module for the transformer layers (self.h).
        4. Use Pipe to parallelize the transformer layers.
        '''

        # BEGIN SOLUTION
        self.pipeline_parallel = True
        
        sequential_layers = []
        for block in self.h:
            if isinstance(block, WithDevice):
                wrapped_block = GPT2BlockWrapper(block.module)
                sequential_layers.append(WithDevice(wrapped_block, block.device))
            else:
                sequential_layers.append(GPT2BlockWrapper(block))

        final_ln_device = _retrieve_device(self.ln_f)
        sequential_layers.append(WithDevice(self.ln_f, final_ln_device))

        transformer_sequential = nn.Sequential(*sequential_layers)

        pipe = Pipe(transformer_sequential, split_size=split_size)
        
        self.h_pp = pipe
        
        
        self.h = nn.ModuleList()        
        # END SOLUTION
        


class GPT2LMHeadModelParallel(GPT2LMHeadModelCustom):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config, GPT2ModelParallel(config))

    def _prepare_pipeline_parallel(self, split_size=1):
        self.parallelize()
        self.transformer._prepare_pipeline_parallel(split_size)

    def _finalize_pipeline_parallel(self):
        self.deparallelize()
        self.transformer.pipeline_parallel = False

if __name__ == '__main__':
    config = AutoConfig.from_pretrained('gpt2')
    model = GPT2LMHeadModelParallel(config=config).to('cuda:0')
    model._prepare_pipeline_parallel()