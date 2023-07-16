from torch import nn
from torchvision.models import mobilenet_v2


class MobileNetV2():
    def __init__(self,weights='IMAGENET1K_V1'):
        self.model = nn.Sequential(*(
            list(
                mobilenet_v2(weights=weights).children()
                )[:-1]
            ))