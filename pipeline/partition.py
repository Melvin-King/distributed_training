from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import nn

class WithDevice(nn.Module):
    def __init__(self, module: nn.Module, device: torch.device):
        super().__init__()
        self._module = module
        self._device = torch.device(device)

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    @property
    def module(self):
        return self._module

    @property
    def device(self):
        return self._device

def _retrieve_device(module: nn.Module) -> torch.device:
    device = None
    for parameter in module.parameters():
        if device is None:
            device = parameter.device
        elif device != parameter.device:
            raise ValueError(
                f'nn.Module: {module}, should have all parameters on a single device,'
                ' please use .to() to place the module on a single device')

    return device if device is not None else torch.device("cpu")

def _assemble_partition(modules: List[nn.Module]):
    modules_list: List[nn.Module] = []
    for module in modules:
        if isinstance(module, nn.Sequential):
            modules_list.extend(module.children())
        else:
            modules_list.append(module)
    return nn.Sequential(*modules_list)

# Homework 2
def _split_module(modules: nn.Sequential) -> Tuple[List[nn.Sequential], List[torch.device]]:
    '''
    Split an nn.Sequential module into partitions and devices.
    '''
    partitions = []
    devices = []

    current_partition = []
    current_device = None
    for name, module in modules.named_children():
        
        # Please read this function and comment the next line of code
        # Q2.1 Comments: This function is to achieve model's pipeline parallelism partitioning. 
        # It goes through all all the modules in nn.sequential, and examines the target device this module should be on.
        # It maintains a current_partition & current_devicie. If a module's target_device is not the same as current_device, then there is a device boundary.
        # In the device boundary, it merges modules in curreent_partition into a nn.sequential module, and add it to partition list as a completed partition. Then it starts a new partition.
        #raise NotImplementedError("Please read the _split_module function.")
        
        if isinstance(module, WithDevice):
            device = module.device
            module = module.module
            module.to(device)
        else:
            device = _retrieve_device(module)
        
        if current_device is not None and current_device != device:
            partitions.append(_assemble_partition(current_partition))
            devices.append(current_device)
            current_partition = []
        
        current_partition.append(module)
        current_device = device

    if current_device is not None:
        partitions.append(_assemble_partition(current_partition))
        devices.append(current_device)

    partitions = nn.ModuleList(partitions)

    return partitions, devices
