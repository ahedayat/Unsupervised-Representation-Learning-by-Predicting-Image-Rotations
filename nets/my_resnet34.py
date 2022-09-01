import torch.nn as nn
from torchvision.models import resnet34

from .nets_utlis import load_net as load


class MyResNet34(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self,
                 num_classes,
                 pretrained=True,
                 dropout=0.2,
                 freeze_backbone=False,
                 ckpt_backbone_path=None,
                 feature_layer_index=4
                 ) -> None:
        super().__init__()

        self.backbone = resnet34(pretrained=pretrained)

        self.num_classes = num_classes

        # Disable Backbone's Classification Layer
        # embedding_size = self.backbone.fc.in_features
        embedding_size = 32 * (2**feature_layer_index)
        self.backbone.fc = nn.Identity()

        if ckpt_backbone_path is not None:
            print(f"Loading '{ckpt_backbone_path}'")
            self.backbone, _ = load(
                ckpt_path=ckpt_backbone_path,
                model=self.backbone,
                optimizer=None
            )
            print("Model is loaded successfully.")

        assert feature_layer_index in [
            1, 2, 3, 4], "Feature layer index must be one of this element: [1,2,3,4]."

        if feature_layer_index < 4:
            self.backbone.layer4 = nn.Identity()
        if feature_layer_index < 3:
            self.backbone.layer3 = nn.Identity()
        if feature_layer_index < 2:
            self.backbone.layer2 = nn.Identity()
        if feature_layer_index < 1:
            self.backbone.layer1 = nn.Identity()

        # self.backbone.avgpool = nn.AdaptiveMaxPool2d((512, 1))

        self.backbone.fc = nn.Identity()

        self.my_classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size//2),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(embedding_size//2, embedding_size//4),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(embedding_size//4, self.num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, _in):
        """
            Forwarding input to output
        """
        out = self.backbone(_in)

        out = self.my_classifier(out)
        # out = self.softmax(out)

        return out

    def freeze_backbone(self):
        """
            Freezing Backbone Network
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
            Unfreezing Backbone Network
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
