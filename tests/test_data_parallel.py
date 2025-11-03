import sys
import os
import pytest
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(cousin_dir))

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

from data_parallel.dataset import partition_dataset
from torch import nn
import torch


@pytest.mark.q1
def test_gradient_accumulation():
    weight0 = torch.load(f"{current_dir}/model0_gradients.pth")
    weight1 = torch.load(f"{current_dir}/model1_gradients.pth")

    assert len(weight0) == len(weight1)
    assert weight0.keys() == weight1.keys()
    for key in weight0.keys():
        assert torch.sum(weight0[key] != weight1[key]) == 0, f"No sync on gradient {key}"
