import torch
import torch.nn as nn
import torch.nn.functional as F


# L2 Pooled
class L2Pool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, eps=1e-12):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps

    def forward(self, x):
        # Square the input
        x_sq = x * x

        # Average pooling over the squared values
        x_sq_pooled = F.avg_pool2d(
            x_sq,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # sqrt of the pooled sum (L2 norm)
        return torch.sqrt(x_sq_pooled + self.eps)


# --------------------------
# Basic 1x1 / 3x3 / 5x5 conv
# --------------------------
class BasicConv(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_c, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


# --------------------------
# Inception block (FaceNet NN2)
# --------------------------
class Inception(nn.Module):
    def __init__(
        self, in_c,
        c1x1,
        c3x3r, c3x3,
        c5x5r, c5x5,
        pool_proj=None,
        strides=None,
        pool_type=None
    ):
        super().__init__()
        if strides is None:
          self.strides = [1,1,1]
        else:
          self.strides = strides

        self.pool_type=pool_type

        # Branch 1: 1x1
        if c1x1 > 0:
            self.b1 = BasicConv(in_c, c1x1, kernel_size=1)
        else:
            self.b1 = None

        # Branch 2: 1x1 → 3x3
        self.b2 = nn.Sequential(
            BasicConv(in_c, c3x3r, kernel_size=1),
            BasicConv(c3x3r, c3x3, kernel_size=3, stride=self.strides[0], padding=1)
        )

        # Branch 3: 1x1 → 5x5
        if c5x5 > 0:
            self.b3 = nn.Sequential(
                BasicConv(in_c, c5x5r, kernel_size=1),
                BasicConv(c5x5r, c5x5, kernel_size=5, stride=self.strides[1], padding=2)
            )
        else:
            self.b3 = None
        
        # Branch 4: MaxPool → 1x1
        if pool_type == 'max':
            self.b4_pool = nn.MaxPool2d(3, stride=self.strides[2], padding=1)
            self.b4 = BasicConv(in_c, pool_proj if pool_proj is not None else in_c, kernel_size=1)
        elif pool_type == 'l2':
            self.b4_pool = L2Pool2d(3, stride=self.strides[2], padding=1)
            self.b4 = BasicConv(in_c, pool_proj if pool_proj is not None else in_c, kernel_size=1)
        else:
            self.b4 = None

    def forward(self, x):
        b1 = self.b1(x) if self.b1 is not None else None
        b2 = self.b2(x)
        b3 = self.b3(x) if self.b3 is not None else None
        b4 = self.b4(self.b4_pool(x)) if self.b4 is not None else None

        return torch.cat([x for x in [b1, b2, b3, b4] if x is not None], dim=1)


# --------------------------
# FaceNet NN2
# --------------------------
class FaceNetNN2(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # 1) conv1 (7×7) -- d=1
        self.conv1 = BasicConv(3, 64, kernel_size=7, stride=2, padding=3)

        # 2) max pool + norm
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(5)

        # 3) inception(2)
        self.inception2 = Inception(
          64, 0, 64, 192, 0, 0, pool_type=None
        )

        # 4) norm + max pool
        self.norm2 = nn.LocalResponseNorm(5)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # 5) inception(3a) — max
        self.inception3a = Inception(
            192, 64, 96, 128, 16, 32, 32, pool_type='max'
        )

        # 6) inception(3b) — L2
        self.inception3b = Inception(
            256, 64, 96, 128, 32, 64, 64, pool_type='l2'
        )

        # 6) inception(3c) — L2
        self.inception3c = Inception(
            320, 0, 128, 256, 32, 64, 320, strides=[2,2,2], pool_type='max'
        )

        # 8) inception(4a) — max
        self.inception4a = Inception(
            640, 256, 96, 192, 32, 64, 128, pool_type='l2'
        )

        # 9) inception(4b) — max
        self.inception4b = Inception(
            640, 224, 112, 224, 32, 64, 128, pool_type='l2'
        )

        # 10) inception(4c) — L2
        self.inception4c = Inception(
            640, 192, 128, 256, 32, 64, 128, pool_type='l2'
        )

        # 11) inception(4d) — max
        self.inception4d = Inception(
            640, 160, 144, 288, 32, 64, 128, pool_type='l2'
        )

        # 12) inception(4e) — L2
        self.inception4e = Inception(
            640, 0, 160, 256, 64, 128, strides=[2,2,2], pool_type='max'
        )

        # 14) inception(5a) — max
        self.inception5a = Inception(
            1024, 384, 192, 384, 48, 128, 128, pool_type='l2'
        )

        # 15) inception(5b) — max
        self.inception5b = Inception(
            1024, 384, 192, 384, 48, 128, 128, pool_type='max'
        )

        # 16) avg pool (7×7 → 1×1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 17) fully connected
        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        # Table 2 — EXACT sequence

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.inception2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        # Final L2-normalization (FaceNet embedding)
        x = F.normalize(x, p=2, dim=1)

        return x


if __name__ == '__main__':
    facenet = FaceNetNN2()
    dummy_data = torch.randn(1, 3, 224, 224)
    output = facenet(dummy_data)
    print(output.shape)
