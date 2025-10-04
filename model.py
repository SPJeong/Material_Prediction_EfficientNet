##### model.py

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class EfficientNetB0(nn.Module):
    def __init__(self, descriptor_size=193):
        super(EfficientNetB0, self).__init__()

        efficientnet_base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Get the number of features from the final convolutional block
        num_ftrs = efficientnet_base.classifier[
            1].in_features  # Access the Linear layer input # 1280 for EfficientNet-B0

        # Create the feature extractor by removing the classifier block (efficientnet has three modules:features, avgpool, classifier)
        self.efficientnet_features = efficientnet_base.features
        self.efficientnet_avgpool = efficientnet_base.avgpool

        # Freeze the EfficientNet-B0 feature layers
        for param in self.efficientnet_features.parameters():
            param.requires_grad = False

        # A separate fc_layers that takes the combined features
        self.fc_layers = nn.Sequential(nn.Linear(num_ftrs + descriptor_size, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 1024),
                                       nn.ReLU(),
                                       nn.Linear(1024, 1024),
                                       nn.Dropout(p=0.3),
                                       nn.ReLU(),
                                       nn.Linear(1024, 1))

    def forward(self, x, descriptors):
        efficientnet_features = self.efficientnet_features(x)
        efficientnet_features = self.efficientnet_avgpool(
            efficientnet_features)  # Apply the final average pooling layer
        efficientnet_features = torch.flatten(efficientnet_features, 1)
        combined_features = torch.cat((efficientnet_features, descriptors), dim=1)
        outputs = self.fc_layers(combined_features)

        return outputs


# # Check the corrected model shape
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = EfficientNetB0()
# model.to(device)
# print(model)
# x = torch.randn(2, 3, 224, 224).to(device)
# d = torch.randn(2, 193).to(device)
# print("Output shape:", model(x, descriptors=d).shape)


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = EfficientNetB0()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")