from copy import Error
import paddle.nn as nn
import paddle

from ._layers_lrp import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
    'resnet101': ('https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
                  '02f35f034ca3858e1e54d4036443c92d'),
    'resnet152': ('https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
                  '7ad16a2f1e7333859ff986138630fd7a'),
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias_attr=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.clone = Clone()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2D(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU()
        self.relu2 = ReLU()

        self.add = Add()

        self.register_forward_post_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.relu2(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu2.relprop(R, alpha)
        out, x2 = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x2 = self.downsample.relprop(x2, alpha)

        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return self.clone.relprop([x1, x2], alpha)

    def m_relprop(self, R, pred, alpha):
        out = self.relu2.m_relprop(R, pred, alpha)
        out, x2 = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x2 = self.downsample.m_relprop(x2, pred, alpha)

        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.relu1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)

        return self.clone.m_relprop([x1, x2], pred, alpha)

    def RAP_relprop(self, R):
        out = self.relu2.RAP_relprop(R)
        out, x2 = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x2 = self.downsample.RAP_relprop(x2)

        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.relu1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return self.clone.RAP_relprop([x1, x2])


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.clone = Clone()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2D(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2D(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2D(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()

        self.add = Add()

        self.register_forward_post_hook(forward_hook)

    def forward(self, x):
        # x1, x2 = self.clone(x, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        # out = self.add([out, x2])
        out = self.add([out, x])
        out = self.relu3(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu3.relprop(R, alpha)

        out, x = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.relu2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x
        # return self.clone.relprop([x1, x2], alpha)

    def m_relprop(self, R, pred, alpha):
        out = self.relu3.m_relprop(R, pred, alpha)

        out, x = self.add.m_relprop(out, pred, alpha)

        if self.downsample is not None:
            x = self.downsample.m_relprop(x, pred, alpha)

        out = self.bn3.m_relprop(out, pred, alpha)
        out = self.conv3.m_relprop(out, pred, alpha)

        out = self.relu2.m_relprop(out, pred, alpha)
        out = self.bn2.m_relprop(out, pred, alpha)
        out = self.conv2.m_relprop(out, pred, alpha)

        out = self.relu1.m_relprop(out, pred, alpha)
        out = self.bn1.m_relprop(out, pred, alpha)
        x1 = self.conv1.m_relprop(out, pred, alpha)
        if paddle.is_tensor(x1) == True:
            return x1 + x
        else:
            for i in range(len(x1)):
                x1[i] = x1[i] + x[i]
            return x1

    def RAP_relprop(self, R):
        out = self.relu3.RAP_relprop(R)

        out, x = self.add.RAP_relprop(out)

        if self.downsample is not None:
            x = self.downsample.RAP_relprop(x)

        out = self.bn3.RAP_relprop(out)
        out = self.conv3.RAP_relprop(out)

        out = self.relu2.RAP_relprop(out)
        out = self.bn2.RAP_relprop(out)
        out = self.conv2.RAP_relprop(out)

        out = self.relu1.RAP_relprop(out)
        out = self.bn1.RAP_relprop(out)
        x1 = self.conv1.RAP_relprop(out)

        return x1 + x


class ResNet(nn.Layer):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                            bias_attr=False)
        self.bn1 = BatchNorm2D(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2D((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

        # for m in [self] + self.sublayers():
        #     if isinstance(m, nn.Conv2D):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2D):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in [self] + self.sublayers():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x

    def relprop(self, R, alpha):
        R = self.fc.relprop(R, alpha)
        R = R.reshape(self.avgpool.Y.shape)
        R = self.avgpool.relprop(R, alpha)

        R = self.layer4.relprop(R, alpha)
        R = self.layer3.relprop(R, alpha)
        R = self.layer2.relprop(R, alpha)
        R = self.layer1.relprop(R, alpha)

        R = self.maxpool.relprop(R, alpha)
        R = self.relu.relprop(R, alpha)
        R = self.bn1.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)

        return R

    def m_relprop(self, R, pred, alpha):
        R = self.fc.m_relprop(R, pred, alpha)
        if paddle.is_tensor(R) == False:
            for i in range(len(R)):
                R[i] = R[i].reshape_as(self.avgpool.Y)
        else:
            R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.m_relprop(R, pred, alpha)

        R = self.layer4.m_relprop(R, pred, alpha)
        R = self.layer3.m_relprop(R, pred, alpha)
        R = self.layer2.m_relprop(R, pred, alpha)
        R = self.layer1.m_relprop(R, pred, alpha)

        R = self.maxpool.m_relprop(R, pred, alpha)
        R = self.relu.m_relprop(R, pred, alpha)
        R = self.bn1.m_relprop(R, pred, alpha)
        R = self.conv1.m_relprop(R, pred, alpha)

        return R

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.RAP_relprop(R)

        R = self.layer4.RAP_relprop(R)
        R = self.layer3.RAP_relprop(R)
        R = self.layer2.RAP_relprop(R)
        R = self.layer1.RAP_relprop(R)

        R = self.maxpool.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.bn1.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    raise NameError("TODO: fix bugs")
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['resnet18'][0],
            model_urls['resnet18'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    raise NameError("TODO: fix bugs")
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['resnet34'][0],
            model_urls['resnet34'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['resnet50'][0],
            model_urls['resnet50'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['resnet101'][0],
            model_urls['resnet101'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        weight_path = paddle.utils.download.get_weights_path_from_url(
            model_urls['resnet152'][0],
            model_urls['resnet152'][1]
        )
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model
