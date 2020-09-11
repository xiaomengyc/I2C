import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from models.obj_simi import obj_simi


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)


        #Added
        self.cls_fc6 = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
        )
        self.cls_fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True))
        self.cls_fc8 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #fc8

        self.center_feat_bank = nn.Parameter(torch.randn((args.num_classes, 2048)), requires_grad=False)
        self.counter=nn.Parameter(torch.zeros(args.num_classes), requires_grad=False)
        self.aux_cls = nn.Linear(2048, args.num_classes)


        self.loss_local_factor = args.loss_local_factor
        self.local_seed_num = args.local_seed_num
        self.loss_global_factor = args.loss_global_factor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, gt_labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        #Branch 1
        out1, last_feat = self.inference(feat4)
        logits_1 = torch.mean(torch.mean(out1, dim=2), dim=2)
        self.atten_map = self.get_atten_map(out1, gt_labels, True) # normalized attention maps
        self.map1 = out1


        if self.training:
            b, w, h = self.atten_map.size()
            mask = torch.zeros((logits_1.size()[0], w, h)).fill_(0).cuda()
            mask = self.get_mask(mask, self.atten_map)

            obj_inds = mask.new_zeros((b, self.local_seed_num))
            obj_simi.obj_valid_inds(mask, obj_inds)
            obj_inds = obj_inds.long()

            obj_out1 = []
            b, c, w,h = feat4.size()
            for i in range(b):
                obj_out1.append(feat4[i].view(c,-1)[:,obj_inds[i]])
            obj_out1 = torch.stack(obj_out1)

            b, c, n = obj_out1.size()
            try:
                tmp1 = obj_out1.view(-1,2, c, n)[:,0,:,:]
                tmp2 = obj_out1.view(-1,2, c, n)[:,1,:,:]
            except RuntimeError:
                pdb.set_trace()
            tmp1 = tmp1.permute(0,2,1).contiguous().view(-1, c)
            tmp2 = tmp2.permute(0,2,1).contiguous().view(-1, c)
            pixel_loss = F.pairwise_distance(tmp1, tmp2, 2)

            cls_center_feat = obj_out1.view(-1, 2, c, n).mean(dim=3).mean(dim=1)

            return [logits_1, pixel_loss, cls_center_feat]
        else:
            pixel_loss = 0
            return [logits_1, self.atten_map, self.get_localization_maps()]
            # return [logits_1, ]


    def inference(self, x):
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc6(x)
        
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc7(x)
        
        if self.training:
            x = F.dropout(x, 0.5)
        out1 = self.cls_fc8(x)

        return out1, x

    def get_localization_maps(self):
        map1 = self.normalize_atten_maps(self.map1)
        return map1


    def mark_obj(self, label_img, heatmap, label, threshold=0.5):

        if isinstance(label, (float, int)):
            np_label = label
        else:
            np_label = label.cpu().data.numpy().tolist()

        for i in range(heatmap.size()[0]):
            mask_pos = heatmap[i] > threshold
            if torch.sum(mask_pos.float()).data.cpu().numpy() < 30:
                threshold = torch.max(heatmap[i]) * 0.7
                mask_pos = heatmap[i] > threshold
            label_i = label_img[i]
            if isinstance(label, (float, int)):
                use_label = np_label
            else:
                use_label = np_label[i]
            # label_i.masked_fill_(mask_pos.data, use_label)
            label_i[mask_pos.data] = use_label
            label_img[i] = label_i

        return label_img

    def mark_bg(self, label_img, heatmap, threshold=0.1):
        mask_pos = heatmap < threshold
        label_img[mask_pos.data] = 0.0

        return label_img

    def get_mask(self, mask, atten_map, th_high=0.7, th_low = 0.05):
        #mask label for segmentation
        mask = self.mark_obj(mask, atten_map, 1.0, th_high)
        mask = self.mark_bg(mask, atten_map, th_low)

        return  mask


    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

    def update_center_vec(self, gt_labels, center_feats):
        batch_size = gt_labels.size(0)
        unique_gt_labels = gt_labels.view(int(batch_size/2), 2)[:,0]

        lr = torch.exp(-0.002*self.counter[unique_gt_labels])
        # lr = torch.exp(-0.001*self.counter[unique_gt_labels])
        fused_centers = (1 - lr.detach()).unsqueeze(dim=1)*self.center_feat_bank[unique_gt_labels].detach() + lr.detach().unsqueeze(dim=1)*center_feats.detach()
        self.counter[unique_gt_labels] += 1
        self.center_feat_bank[unique_gt_labels] = fused_centers
        return fused_centers


    def get_loss(self, logits, gt_labels):
        cls_logits, pixel_loss, batch_center_vecs  = logits
        gt = gt_labels.long()
        loss_cls = F.cross_entropy(cls_logits, gt)

        batch_size = gt_labels.size(0)
        unique_gt_labels = gt_labels.view(int(batch_size/2), 2)[:,0]
        # aux_loss_cls = F.cross_entropy(aux_logits, unique_gt_labels.long())

        aux_loss_cls = F.pairwise_distance(self.center_feat_bank[unique_gt_labels].cuda().detach(), batch_center_vecs, 2)
        self.update_center_vec(gt_labels, batch_center_vecs.detach())


        loss = loss_cls + self.loss_local_factor*pixel_loss.mean() + self.loss_global_factor*aux_loss_cls.mean()
        return loss, loss_cls, self.loss_local_factor*pixel_loss.mean(), self.loss_global_factor*aux_loss_cls.mean()



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def model(pretrained=False, threshold=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
