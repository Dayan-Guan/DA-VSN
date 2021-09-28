import torch
import torch.nn as nn

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.sf_layer = self.get_score_fusion_layer(num_classes)

    def get_score_fusion_layer(self, num_classes):
        sf_layer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(sf_layer.weight)
        nn.init.eye_(sf_layer.weight[:, :num_classes, :, :].squeeze(-1).squeeze(-1))
        return sf_layer

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, cf, kf, flow, device):
        cf = self.conv1(cf)
        cf = self.bn1(cf)
        cf = self.relu(cf)
        cf = self.maxpool(cf)
        cf = self.layer1(cf)
        cf = self.layer2(cf)
        cf = self.layer3(cf)
        if self.multi_level:
            cf_aux = self.layer5(cf)
        else:
            cf_aux = None
        cf = self.layer4(cf)
        cf = self.layer6(cf)
        with torch.no_grad():
            kf = self.conv1(kf)
            kf = self.bn1(kf)
            kf = self.relu(kf)
            kf = self.maxpool(kf)
            kf = self.layer1(kf)
            kf = self.layer2(kf)
            kf = self.layer3(kf)
            if self.multi_level:
                kf_aux = self.layer5(kf)
            else:
                kf_aux = None
            kf = self.layer4(kf)
            kf = self.layer6(kf)
        kf_aux_cpu = kf_aux.cpu().numpy()
        kf_cpu = kf.cpu().numpy()
        interp_flow2cf = nn.Upsample(size=(cf.shape[-2], cf.shape[-1]), mode='bilinear', align_corners=True)
        interp_flow2cf_ratio = cf.shape[-2] / flow.shape[-2]
        flow_cf = interp_flow2cf(flow) * interp_flow2cf_ratio
        flow_cf = flow_cf.cpu().numpy()
        import numpy as np
        kf_aux_rec = np.zeros(cf_aux.shape)
        kf_rec = np.zeros(cf.shape)
        rec_positions = np.zeros(cf.shape)
        for x in range(cf.shape[-1]):
            for y in range(cf.shape[-2]):
                x_flow = int(round(x - flow_cf[:, 0, y, x][0]))
                y_flow = int(round(y - flow_cf[:, 1, y, x][0]))
                if x_flow >= 0 and x_flow < flow_cf.shape[-1] and y_flow >= 0 and y_flow < flow_cf.shape[-2]:
                    kf_aux_rec[:, :, y_flow, x_flow] = kf_aux_cpu[:, :, y, x]
                    kf_rec[:, :, y_flow, x_flow] = kf_cpu[:, :, y, x]
                    rec_positions[:, :, y_flow, x_flow] = 1
        kf_aux_rec = torch.from_numpy(kf_aux_rec)
        kf_rec = torch.from_numpy(kf_rec)
        rec_positions = torch.from_numpy(rec_positions)


        pred_aux = self.sf_layer(torch.cat((cf_aux, (rec_positions*kf_aux_rec).float().cuda(device)), dim=1))
        pred = self.sf_layer(torch.cat((cf, (rec_positions*kf_rec).float().cuda(device)), dim=1))
        return pred_aux, pred, cf_aux, cf, kf_aux, kf

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_sf_layer(self):
        b = []
        b.append(self.sf_layer.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_1x_lr_params_sf_layer(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def get_accel_deeplab_v2(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model
