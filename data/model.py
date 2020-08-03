import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class XrayModel(nn.Module):
    def __init__(self):
        super(XrayModel, self).__init__()

        # cnn containing vgg19 model
        self.cnn = models.vgg19_bn(pretrained=True)

        # Additional layers required to reformat the outputted classification
        self.fc1 = nn.Linear(1010, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, image, gender):
        x1 = self.cnn(image)
        x2 = gender  # Gender is 0 or 1 (female / male respectivly)

        x2 = x2.unsqueeze(1)

        # Reformat gender to be 10x1 instead of 1x1
        # Means it will carry more weight in final classification
        x2 = torch.cat((x2, x2, x2, x2, x2, x2, x2, x2, x2, x2), dim=1)

        # Combine gender with output from the VGG19 network
        x = torch.cat((x1, x2), dim=1)

        # Run combined tensor through final two fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x




