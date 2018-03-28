from torch import nn
from torchvision import models
from utils import get_upsampling_weight
from models.segmentation_model import Model

class FCN32s(Model):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32s, self).__init__()
        # load the pretrained models.
        vgg = models.vgg16(pretrained=pretrained)
        # freeze the pretrained parameters.
        for param in vgg.parameters():
            param.requires_grad = False

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
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
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        )
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32 * 7, stride=32 * 7, bias=False)
        #
        #
        # upscore7 = nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=7, bias=False)
        # upscore7.weight.data.copy_(get_upsampling_weight(1024, 512, 7))
        # upscore32 = nn.ConvTranspose2d(512, num_classes, kernel_size=32, stride=32, bias=False)
        # upscore32.weight.data.copy_(get_upsampling_weight(512, num_classes, 32))
        # self.upscore = nn.Sequential(
        #     fc6, nn.ReLU(inplace=True),
        #     upscore7, nn.ReLU(inplace=True), upscore32
        # )

        print('total number of parameters in feature: ', self.count_parameters(self.features5))
        print('total number of parameters in classif: ', self.count_parameters(self.upscore))

        # score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        # score_fr.weight.data.zero_()
        # score_fr.bias.data.zero_()
        # self.score_fr = nn.Sequential(
        #     fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        # )
        # self.score_fr = nn.Sequential(
        #     fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True), score_fr
        # )
        # self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)

    def forward(self, x):
        pool5 = self.features5(x)
        upscore = self.upscore(pool5)
        # upscore = self.upscore(score_fr)
        # return upscore.contiguous()
        return upscore

    @staticmethod
    def count_parameters(model):
        count = 0
        for param in model.parameters():
            size = 1
            for s in list(param.size()):
                size *= s
            count += size
        return count


def main():
    vgg16 = models.vgg16()
    print(vgg16)


if __name__ == '__main__':
    main()
