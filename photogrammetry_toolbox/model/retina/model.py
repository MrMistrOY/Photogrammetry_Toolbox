import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork, nms
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from photogrammetry_toolbox.model.retina.anchors import Anchors
from photogrammetry_toolbox.model.retina.losses import FocalLoss
from photogrammetry_toolbox.model.retina.utils import BBoxTransform, ClipBoxes


class Regressor(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(Regressor, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class Classifier(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class RetinaNet(nn.Module):
    def __init__(self, backbone: str, pretrained_backbone=True, config={}):
        super(RetinaNet, self).__init__()
        self.num_classes = config.get('num_classes', 1)
        self.top_k = config.get('top_k', 1000)
        if backbone == 'resnet18':
            feature_extractor = resnet.resnet18(pretrained=pretrained_backbone)
        elif backbone == 'resnet34':
            feature_extractor = resnet.resnet34(pretrained=pretrained_backbone)
        elif backbone == 'resnet50':
            feature_extractor = resnet.resnet50(pretrained=pretrained_backbone)
        elif backbone == 'resnet101':
            feature_extractor = resnet.resnet101(pretrained=pretrained_backbone)
        elif backbone == 'resnet152':
            feature_extractor = resnet.resnet152(pretrained=pretrained_backbone)
        elif backbone == 'resnext50_32x4d':
            feature_extractor = resnet.resnext50_32x4d(pretrained=pretrained_backbone)
        elif backbone == 'resnext101_32x8d':
            feature_extractor = resnet.resnext101_32x8d(pretrained=pretrained_backbone)
        elif backbone == 'wide_resnet50_2':
            feature_extractor = resnet.wide_resnet50_2(pretrained=pretrained_backbone)
        elif backbone == 'wide_resnet101_2':
            feature_extractor = resnet.wide_resnet101_2(pretrained=pretrained_backbone)
        else:
            raise NotImplementedError('unknown backbone')

        self.backbone = IntermediateLayerGetter(feature_extractor,
                                                {'layer2': 'p3',
                                                 'layer3': 'p4',
                                                 'layer4': 'p5'})

        # RetinaNetFPN
        if isinstance(self.backbone['layer2'][-1], resnet.BasicBlock):
            in_channels_list = [self.backbone['layer2'][-1].conv2.out_channels,
                                self.backbone['layer3'][-1].conv2.out_channels,
                                self.backbone['layer4'][-1].conv2.out_channels]
        elif isinstance(self.backbone['layer2'][-1], resnet.Bottleneck):
            in_channels_list = [self.backbone['layer2'][-1].conv3.out_channels,
                                self.backbone['layer3'][-1].conv3.out_channels,
                                self.backbone['layer4'][-1].conv3.out_channels]
        else:
            raise NotImplementedError('unknown block type')

        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channels=256,
                                         extra_blocks=LastLevelP6P7(in_channels=256, out_channels=256))

        self.focalLoss = FocalLoss()

        self.regressor = Regressor(256)
        self.classifier = Classifier(256, num_classes=self.num_classes)
        self.anchors = Anchors()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        body = self.backbone(img_batch)
        features = self.fpn(body)

        regression = torch.cat([self.regressor(feature) for (level, feature) in features.items()], dim=1)
        classification = torch.cat([self.classifier(feature) for (level, feature) in features.items()], dim=1)

        anchors = self.anchors(img_batch)
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            batch_size = classification.shape[0]

            finalScores = torch.zeros(batch_size, self.top_k).fill_(value=-1.0)
            finalAnchorBoxesIndexes = torch.zeros(batch_size, self.top_k).fill_(value=-1).long()
            finalAnchorBoxesCoordinates = torch.zeros(batch_size, self.top_k, 4).fill_(value=-1.0)

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
            for b in range(batch_size):
                # b for batch
                b_scores = torch.tensor([]).float().cuda()
                b_labels = torch.tensor([]).long().cuda()
                b_boxes = torch.tensor([]).float().cuda()
                for c in range(batch_size):
                    # c for each class
                    scores = classification[b, :, c]
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                        # no boxes to NMS, just continue
                        continue

                    scores = scores[scores_over_thresh]
                    anchorBoxes = transformed_anchors[b]
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                    b_scores = torch.cat([b_scores, scores[anchors_nms_idx]])
                    b_labels = torch.cat(
                        [b_labels, torch.zeros(anchors_nms_idx.shape[0], dtype=torch.long).cuda().fill_(value=c)])
                    b_boxes = torch.cat([b_boxes, anchorBoxes[anchors_nms_idx]])
                b_scores = b_scores[:self.top_k]
                b_labels = b_labels[:self.top_k]
                b_boxes = b_boxes[:self.top_k]

                n_det = b_scores.shape[0]
                if n_det > 0:
                    finalScores[b, :n_det] = b_scores
                    finalAnchorBoxesIndexes[b, :n_det] = b_labels
                    finalAnchorBoxesCoordinates[b, :n_det] = b_boxes

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def load(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self.load_state_dict(state_dict)
