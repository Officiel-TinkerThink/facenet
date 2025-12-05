import torchvision.datasets import CelebA
dataset = CelebA(root="./.data", split='train', download=True)