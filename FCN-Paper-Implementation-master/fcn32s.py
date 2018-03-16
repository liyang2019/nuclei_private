from torch import nn
from torchvision import models
from utils import get_upsampling_weight


class FCN32s(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32s, self).__init__()
        # load the pretrained model.
        vgg = models.vgg16(pretrained=pretrained)
        # freeze the pretrained parameters.
        for param in vgg.parameters():
            param.requires_grad = False

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        # self.score_fr = nn.Sequential(
        #     fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        # )
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        # # do not update the parameters of the fully connected layer?
        # for param in self.score_fr.parameters():
        #     param.requires_grad = False

        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))

    def forward(self, x):
        m, c, h, w = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19: (19 + h), 19: (19 + w)].contiguous()


def main():
    vgg16 = models.vgg16()
    print(vgg16)


if __name__ == '__main__':
    main()
