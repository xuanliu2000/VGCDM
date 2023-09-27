import torch
from torch import nn
from utils.attention import SpatialSelfAttention

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input 1824
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            SpatialSelfAttention(256),

            nn.Conv1d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 1, kernel_size=128, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.convT1 = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 64, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        self.convT2 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True))
        self.convT3 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True))

        self.att=SpatialSelfAttention(128)

        self.convT4 = nn.Sequential(

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True))

        self.convT5 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x=self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x= self.att(x)
        x = self.convT4(x)
        x = self.convT5(x)
        return x


if __name__ == "__main__":
    batch_size = 64


    noise = torch.randn(batch_size, 256,1)
    G = Generator(256)
    D = Discriminator()
    fake_data = G(noise)
    print(f"Shape of generated data: {fake_data.shape}")
    decision = D(fake_data)
    print(f"Shape of discriminator decision: {decision.shape}")

