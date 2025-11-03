from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# Homework 2
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        # Please read this function and comment the next line of code
        # This is a view class and wraps the original dataset. It only exposes a given index mapping to a subset. 
        # It aims to make DataLoader work like ordinary datasets, but only allows access to samples distributed to a partition.
        #raise NotImplementedError("Please read the Partition class.")
        return self.data[index]

# Homework 2
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)

        # Please read this function and comment the next line of code
        # This is in charge of shuffling the index of the entire dataset and splitting it into different partitions. 
        # This class stores each partition’s index list, and we can use use(rank) to obtain a specific partition object. 
        # This enables us to divide the dataset into different GPUs, and each GPU calls use() to obtain their own partitions, respectively.
        #raise NotImplementedError("Please read the DataPartitioner class.")
    
        data_len = len(data)
        indices = list(range(data_len))
        rng.shuffle(indices)

        for size in sizes:
            partition_len = int(size * data_len)
            self.partitions.append(indices[:partition_len])
            indices = indices[partition_len:]

    def use(self, partition):
        
        # Please read this function and comment the next line of code
        #raise NotImplementedError("Please read the use function.")
        return Partition(self.data, self.partitions[partition])

# Homework 2
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ 
    Partitioning training dataset of the Machine Translation
    """
    
    # Please read this function and comment the next line of code
    # The function is to achieve the shard of the dataset and make every GPU to only see a part of the dataset and 
    # compute its own gradient for later parameter synchronization.
    #raise NotImplementedError("Please read the partition_dataset function.")

    partition_size = batch_size // world_size
    partitioner = DataPartitioner(dataset, sizes = [1 / world_size] * world_size )
    partitioned_data = partitioner.use(rank)
    dataloader = DataLoader(partitioned_data, batch_size=partition_size, collate_fn=collate_fn, shuffle=True)
    return dataloader

    #

