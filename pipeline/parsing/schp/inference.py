import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from pipeline.parsing.schp.preprocess import preprocess_image
from pipeline.parsing.schp.postprocess import postprocess_logits
from pipeline.config import settings

class ParsingResult:
    """
    Standardized container holding output from the SCHP Semantic human parser.
    """
    def __init__(
        self, 
        segmentation_map: np.ndarray, 
        confidence_map: np.ndarray, 
        label_masks: Dict[str, np.ndarray]
    ):
        self.segmentation_map = segmentation_map  # np.ndarray of shape (H, W) label indices
        self.confidence_map = confidence_map      # np.ndarray of shape (H, W) probabilities
        self.label_masks = label_masks            # Dict[str, np.ndarray] binary masks


# --- CE2P Human Parsing Network Architecture ---

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, abn=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
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

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing Module (PPM) matching context_encoding checkpoint keys.
    """
    def __init__(self, features=2048, out_features=512, sizes=(1, 2, 3, 6), abn_class=None):
        super(PSPModule, self).__init__()
        if abn_class is None:
            from pipeline.parsing.schp.model_loader import MockInPlaceABN
            abn_class = MockInPlaceABN

        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size, abn_class) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, bias=False),
            abn_class(out_features),
        )

    def _make_stage(self, features, out_features, size, abn_class):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = abn_class(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class EdgeModule(nn.Module):
    """
    Edge Perceiving Branch matching edge checkpoint keys.
    """
    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2, abn_class=None):
        super(EdgeModule, self).__init__()
        if abn_class is None:
            from pipeline.parsing.schp.model_loader import MockInPlaceABN
            abn_class = MockInPlaceABN

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, bias=False),
            abn_class(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, bias=False),
            abn_class(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, bias=False),
            abn_class(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class DecoderModule(nn.Module):
    """
    Parsing Branch Decoder Module matching decoder checkpoint keys.
    """
    def __init__(self, num_classes, abn_class=None):
        super(DecoderModule, self).__init__()
        if abn_class is None:
            from pipeline.parsing.schp.model_loader import MockInPlaceABN
            abn_class = MockInPlaceABN

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
            abn_class(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, bias=False),
            abn_class(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, bias=False),
            abn_class(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            abn_class(256)
        )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class ResNet101_CE2P(nn.Module):
    """
    Official ResNet-101 backbone with CE2P parsing and fusion heads.
    Aligns 100% with the Hugging Face pre-trained exp-schp weight checkpoints.
    """
    def __init__(self, num_classes=20, abn_class=None):
        super(ResNet101_CE2P, self).__init__()
        if abn_class is None:
            from pipeline.parsing.schp.model_loader import MockInPlaceABN
            abn_class = MockInPlaceABN

        self.inplanes = 128
        
        # ResNet Backbone conv stem
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layer stages
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=2)

        # Head modules aligning with the checkpoint keys
        self.context_encoding = PSPModule(features=2048, out_features=512, abn_class=abn_class)
        self.edge = EdgeModule(in_fea=[256, 512, 1024], mid_fea=256, out_fea=2, abn_class=abn_class)
        self.decoder = DecoderModule(num_classes, abn_class=abn_class)

        # Fusion head merging semantic-aware and edge-aware features
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
            abn_class(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size()[2:]

        # ResNet backbone
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Parsing Branch
        x_context = self.context_encoding(x5)
        _, parsing_fea = self.decoder(x_context, x2)
        
        # Edge Branch
        _, edge_fea = self.edge(x2, x3, x4)
        
        # Fusion Branch
        x_fuse = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x_fuse)

        # Interpolate final fusion result back to input resolution
        fusion_result = F.interpolate(fusion_result, size=(h, w), mode='bilinear', align_corners=True)
        return fusion_result


# --- Central Inference Executor ---

def run_schp_inference(img: np.ndarray) -> ParsingResult:
    """
    Executes standard pre-processing, PyTorch singleton model evaluation,
    and post-processing for a centered human crop image.
    """
    from pipeline.parsing.schp.model_loader import schp_model_loader
    
    # 1. Retrieve the PyTorch model
    model = schp_model_loader.get_model()
    
    # 2. Preprocess image crop
    h, w = img.shape[:2]
    tensor = preprocess_image(img, target_size=473)
    
    # Move to GPU/CPU device
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    # 3. Model inference pass
    with torch.no_grad():
        logits = model(tensor)
        
    # 4. Postprocess logits back to original crop resolution
    seg_map, conf_map, label_masks = postprocess_logits(
        logits=logits, 
        original_size=(w, h), 
        confidence_threshold=settings.schp_confidence_threshold
    )
    
    return ParsingResult(seg_map, conf_map, label_masks)
