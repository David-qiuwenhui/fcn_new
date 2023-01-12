'''
coding: utf-8
@Author: qiuwenhui
@Software: PyCharm
@Time: 2023/1/9 17:34
'''
from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .resnet_backbone import resnet50, resnet101
from .mobilenet_backbone import mobilenet_v3_large


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):  # 进行关系测试，检测return_layers是否是model模块的子集
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if (
                name in self.return_layers
            ):  # return_layer = {dict: 2} {'layer4':'out', 'layer3':''aux}
                out_name = self.return_layers[name]
                out[out_name] = x
        return out  # out = {OrderedDict:2} {'aux':Tensor(bs,1024,60,60), 'out':Tensor{bs,2048,60,60}}


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    __constants__ = ["aux_classifier"]

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]  # x(bs,3,480,480)
        
        # contract: features is a dict of tensors
        features = self.backbone(x)  # feature={OrderedDict:2} {'aux':Tensor(bs,1024,60,60),'out':Tensor(bs,2048,60,60)}
        
        # 输出特征提取中主分类器和辅助分类器的特征信息
        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)  # x(bs,21,60,60)
        # 使用双线性插值还原回原图尺度
        # DeepLabV3是在Encoder后进行单阶段的特征上采样
        x = F.interpolate(
            input=x, size=input_shape, mode="bilinear", align_corners=False
        )  # x(bs,21,480,480)
        result["out"] = x
        # 辅助分类器 Decoder
        if self.aux_classifier is not None:
            x = features["aux"]  # x(bs,1024,60,60)
            x = self.aux_classifier(x)  # x(bs,21,60,60)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(
                input=x, size=input_shape, mode="bilinear", align_corners=False
            )  # x(bs,21,480,480)
            result[
                "aux"
            ] = x  # result={OrderedDict:2} {'out':Tensor(bs,21,480,480),'aux':Tensor(bs,21,480,480)}

        # return result
        return (result["out"], result["aux"])


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4  # inter_channels=1024//4=256 辅助分支中间部分的维度
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(
                inter_channels, channels, kernel_size=1
            ),  # channels为模型的分类类别个数 使用1x1的卷积核进行维度调整
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]  # size=torch.Size([60, 60])
        for mod in self:
            x = mod(x)
        return F.interpolate(
            input=x, size=size, mode="bilinear", align_corners=False
        )  # 通过AdaptiveAvgPool模块处理后的feature maps (h,w)=(1,1)，
        # 对feature maps进行双线性插值上采样到(60, 60)，即用同一个元素填充feature maps 2d到(60, 60)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False
                ),  # nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
                nn.BatchNorm2d(out_channels),  # out_channels=256
                nn.ReLU(),
            )
        ]  # 1.Conv1x1卷积模块
        rates = tuple(atrous_rates)  # rates={tuple:3}(12,24,36)
        for rate in rates:
            modules.append(
                ASPPConv(in_channels, out_channels, dilation=rate)
            )  # 2.Conv3x3(r12)模块，3.Conv3x3(r24)模块，4.Conv3x3(r36)模块

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        # self.convs = {ModuleList:5} {(0)Sequential, (1)ASPPConv, (2)ASPPConv, (3)ASPPConv, (4)ASPPPooling}
        self.project = nn.Sequential(
            nn.Conv2d(
                len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:  # 进入五个ASPP分支进行膨胀卷积处理
            _res.append(conv(x))
        res = torch.cat(
            _res, dim=1
        )  # 将ASPP模块的五个分支在dim=1上进行concat拼接 5 x (bs,256,60,60) -> (bs,1280,60,60)
        return self.project(res)  # 对拼接后的结果再进行Conv1x1调整维度，BN，ReLU和Dropout处理


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates=[12, 24, 36]),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )


def deeplabv3_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'
    backbone = resnet50(
        replace_stride_with_dilation=[False, True, True]
    )  # 下采样率为8倍，layer0/1/2进行下采样，layer3/4进行膨胀卷积

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(
            torch.load("./pre-weights/resnet50_imagenet.pth", map_location="cpu")
        )

    out_inplanes = 2048  # 输入主分类器的feature maps channels
    aux_inplanes = 1024  # 输入辅助分类器的feature maps channels

    return_layers = {"layer4": "out"}  # 返回layer4的输出给classifier
    if aux:
        return_layers["layer3"] = "aux"  # 返回layer3的输出给aux_classifier
    # 重新构造backbone
    backbone = IntermediateLayerGetter(model=backbone, return_layers=return_layers)

    # 辅助分类器
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(
            in_channels=aux_inplanes, channels=num_classes
        )  # aux_inplanes=1024

    # 主分类器 特征提取主干分支
    classifier = DeepLabHead(
        in_channels=out_inplanes, num_classes=num_classes
    )  # out_inplanes=2048

    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


def deeplabv3_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location="cpu"))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)

    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


def deeplabv3_mobilenetv3_large(aux, num_classes=21, pretrain_backbone=False):
    # 'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    # 'depv3_mobilenetv3_large_coco': "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth"
    backbone = mobilenet_v3_large(dilated=True)

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        backbone.load_state_dict(
            torch.load("mobilenet_v3_large.pth", map_location="cpu")
        )

    backbone = backbone.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)]
        + [len(backbone) - 1]
    )
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)

    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

