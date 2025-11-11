import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

# VGG16 Encoder (Backbone)
class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg16.features.children())
        self.block1 = nn.Sequential(*features[0:5])    
        self.block2 = nn.Sequential(*features[5:10])  
        self.block3 = nn.Sequential(*features[10:17]) 
        self.block4 = nn.Sequential(*features[17:24]) 
        self.block5 = nn.Sequential(*features[24:31])  

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)
        return f1, f2, f3, f4, f5

# FCN8 Decoder
class FCN8Decoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        hidden_channels = 4096

        self.conv6 = nn.Conv2d(512, hidden_channels, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.score_fr = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)

        self.deconv1 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=8, stride=8, padding=0, bias=False)

        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4, f5 = features

        x = self.relu6(self.conv6(f5))
        x = self.relu7(self.conv7(x))
        x = self.score_fr(x)

        o = self.deconv1(x)
        target_size = self.score_pool4(f4).shape[2:]
        o = center_crop(o, target_size)

        o2 = self.score_pool4(f4)
        o = o + o2

        o = self.deconv2(o)
        target_size = self.score_pool3(f3).shape[2:]
        o = center_crop(o, target_size)

        o2 = self.score_pool3(f3)
        o = o + o2

        o = self.deconv3(o)

        return o  # raw logits

# Center crop helper
def center_crop(tensor, target_size):
    _, _, h, w = tensor.shape
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[:, :, i:i+th, j:j+tw]

# Final FCN8s model
class FCN8s(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.encoder = VGG16Backbone()
        self.decoder = FCN8Decoder(n_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out  # raw logits
