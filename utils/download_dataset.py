from torchvision import datasets
dataset = datasets.CelebA(root="./.data", split='train', download=True)