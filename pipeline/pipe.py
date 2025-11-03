from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# Homework 2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''
    Generate schedules for each clock cycle.

    m: number of micro-batches
    n: number of partitions
    i: index of micro-batch
    j: index of partition
    k: clock number
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function yields schedules for each clock cycle.
    '''


    # BEGIN SOLUTION
    m = num_batches
    n = num_partitions
    
    k_total = m + n - 1
    
    for k in range(k_total):
        schedule: List[Tuple[int, int]] = []
        
        
        for i in range(m):
            j = k - i
            
            if 0 <= j < n:
                schedule.append((i, j))
        yield schedule
    #raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # Homework 2
    def forward(self, x):
        ''' 
        Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.

        '''

        # BEGIN SOLUTION
        split_size = self.split_size
        
        batches = list(torch.split(x, split_size, dim=0))
        num_batches = len(batches)
        num_partitions = len(self.partitions)

        schedule_generator = _clock_cycles(num_batches, num_partitions)

        for schedule in schedule_generator:
            if schedule:
                self.compute(batches, schedule)
        final_device = self.devices[-1]
        
        final_output = torch.cat(
            [batch.to(final_device) for batch in batches], dim=0
        )
        
        return final_output
        #raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
        # END SOLUTION

    # Homework 2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''
        Compute the micro-batches in parallel.
        '''
        
        partitions = self.partitions
        devices = self.devices

        # Please read this function and comment the next line of code
        # Q2.2 comments: This function is to achieve the core of compute and communication in pipeline parallelism.

        # It receives 2 parameters of batches(list of all mocre batches) and schedule(schedule list for current clock cycle).
        # It send tasks to Worker by traversal schedule list and moves micro batchs that need to be computed to the device of the target partition.
        # Then, it creates a Task object and put it into the corresponding in_queues of target partition.
        # Workers will pick out these tasks and execute them.

        # Then, it traversal all the (micro_batch, partition_idx) pairs in schedule list.
        # And it takes computation results from out_queues of target partition. This step achieves sync wait of computation completion (worker is async).
        # If cmputation is successful, then takes out new result[1] and use it to update batches[micro_batch], and result[1] will be used as input of the next cycle/partition.
        #raise NotImplementedError("Please read the compute funtion")
    
        for micro_batch, partition_idx in schedule:
            # Step 1: Send tasks to the workers based on the schedule
            batches[micro_batch] = batches[micro_batch].to(devices[partition_idx])
            task = Task(partitions[partition_idx], batches[micro_batch])
            self.in_queues[partition_idx].put(task)


        for micro_batch, partition_idx in schedule:
            # Step 2: Retrieve results from the workers
            success, result = self.out_queues[partition_idx].get()
            batches[micro_batch] = result[1]


