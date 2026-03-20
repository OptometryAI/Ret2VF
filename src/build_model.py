import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur



def generate_model(cfg):
    model_type = _cfg_get(cfg, 'model', 'model_type', default='resnet')
    assert model_type in ['resnet']

    if model_type == 'resnet':
        return _generate_resnet_from_cfg(cfg)

    raise ValueError(f"Unsupported model_type: {model_type}")



def _generate_resnet_from_cfg(cfg):
    return generate_resnet(
        model_depth=_cfg_get(cfg, 'model', 'model_depth', default=18),
        task_type=_cfg_get(cfg, 'task', 'task_type', default='c'),
        n_input_channels=_cfg_get(cfg, 'model', 'n_input_channels', default=1),
        n_classes=_cfg_get(cfg, 'model', 'n_classes', default=54),
        n_categories_per_class=_cfg_get(cfg, 'model', 'n_categories_per_class', default=50),
        predict_classes=_cfg_get(cfg, 'multitask', 'predict_classes', default=True),
        conv1_t_size=_cfg_get(cfg, 'model', 'conv1_t_size', default=7),
        conv1_t_stride=_cfg_get(cfg, 'model', 'conv1_t_stride', default=2),
        shortcut_type=_cfg_get(cfg, 'model', 'shortcut_type', default='B'),
    )



def get_inplanes():
    return [64, 128, 256, 512]



def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )



def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x=out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=2,
        shortcut_type='B',
        n_classes=54,
        n_categories_per_class=50,
        task_type='r',
        predict_classes=False,
    ):
        super().__init__()

        self.task_type = task_type
        self.n_classes = n_classes
        self.n_categories_per_class = n_categories_per_class
        self.predict_classes = predict_classes
        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 1, 1),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((128, 72, 72))

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.conv_1x1x1 = nn.Conv3d(block_inplanes[3] * block.expansion, 54, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv3d(54, n_classes, kernel_size=(16, 9, 9), stride=1, padding=0)

        if self.task_type == 'c':
            self.linear_bias = nn.Parameter(torch.zeros(n_classes, n_categories_per_class).float())
        elif self.task_type != 'r':
            raise ValueError(f"Unsupported task_type: {task_type}. Expected 'r' or 'c'.")

        if self.predict_classes:
            self.classification_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.regression_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.classification_fc = nn.Linear(block_inplanes[3] * block.expansion, 2)
            self.md_regression_fc = nn.Linear(block_inplanes[3] * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
        out = torch.cat([out, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        shared_features = x

        main_output = self.conv_1x1x1(x)
        main_output = self.final_conv(main_output)

        if self.task_type == 'r':
            batch_size = main_output.size(0)
            main_output = main_output.squeeze()
            if batch_size == 1 and main_output.dim() < 2:
                main_output = main_output.view(batch_size, *main_output.shape)
        else:
            main_output = main_output.squeeze().unsqueeze(-1)
            main_output = torch.tile(main_output, (1, 1, self.n_categories_per_class))
            main_output = main_output + self.linear_bias

        if not self.predict_classes:
            return main_output

        cls_output = self.classification_pool(shared_features)
        cls_output = cls_output.view(cls_output.size(0), -1)
        cls_output = self.classification_fc(cls_output)

        md_output = self.regression_pool(shared_features)
        md_output = md_output.view(md_output.size(0), -1)
        md_output = self.md_regression_fc(md_output)

        return {
            'main': main_output,
            'pm_classification': cls_output,
            'md_regression': md_output,
        }



def generate_resnet(model_depth, task_type='r', n_classes=400, n_categories_per_class=2, predict_classes=False, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    assert task_type in ['r', 'c']

    model_params = {
        10: {'block': BasicBlock, 'layers': [1, 1, 1, 1]},
        18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
        101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
        152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        200: {'block': Bottleneck, 'layers': [3, 24, 36, 3]},
    }
    params = model_params[model_depth]
    return ResNet(
        block=params['block'],
        layers=params['layers'],
        block_inplanes=get_inplanes(),
        n_classes=n_classes,
        n_categories_per_class=n_categories_per_class,
        task_type=task_type,
        predict_classes=predict_classes,
        **kwargs,
    )
