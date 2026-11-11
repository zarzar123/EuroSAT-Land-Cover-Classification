
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
