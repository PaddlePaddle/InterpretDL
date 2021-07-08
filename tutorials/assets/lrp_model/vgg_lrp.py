import paddle.nn as nn
import paddle

from ._layers_lrp import *

__all__ = [
    'VGG', 'vgg16'
]


model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              '89bbffc0f87d260be9b8cdc169c991c4')
}


class VGG(nn.Layer):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2D((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.classifier(x)
        return x

    def relprop(self, R, alpha):
        x = self.classifier.relprop(R, alpha)
        x = x.reshape(next(reversed(self.features._sub_layers.values())).Y.shape)
        x = self.avgpool.relprop(x, alpha)
        x = self.features.relprop(x, alpha)

        return x
    def m_relprop(self, R, pred, alpha):
        x = self.classifier.m_relprop(R, pred, alpha)
        if paddle.is_tensor(x) == False:
            for i in range(len(x)):
                x[i] = x[i].reshape(next(reversed(self.features._sub_layers.values())).Y.shape)
        else:
            x = x.reshape(next(reversed(self.features._sub_layers.values())).Y.shape)
        x = self.avgpool.m_relprop(x, pred, alpha)
        x = self.features.m_relprop(x, pred, alpha)

        return x
    def RAP_relprop(self, R):
        x1 = self.classifier.RAP_relprop(R)
        if paddle.is_tensor(x1) == False:
            for i in range(len(x1)):
                x1[i] = x1[i].reshape(next(reversed(self.features._sub_layers.values())).Y.shape)
        else:
            x1 = x1.reshape(next(reversed(self.features._sub_layers.values())).Y.shape)
        x1 = self.avgpool.RAP_relprop(x1)
        x1 = self.features.RAP_relprop(x1)

        return x1


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2D(v), ReLU()]
            else:
                layers += [conv2d, ReLU()]
            in_channels = v
    return Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['vgg16'][0],
            model_urls['vgg16'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model

