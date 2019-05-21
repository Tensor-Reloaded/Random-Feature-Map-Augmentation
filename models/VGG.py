import torch.nn as nn
import torchvision.transforms as transforms
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.transf_prob = 0.1
        self.pre_procces  = transforms.Compose([
            transforms.ToPILImage(),
        ])
        self.post_procces = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transf = transforms.RandomChoice([
            transforms.RandomRotation(10),
            # transforms.RandomCrop(32, 4),
        ])

        self.complete_transf = transforms.Compose([self.pre_procces, self.transf, self.post_procces])
    def forward(self, x, train = False):
        device = torch.device('cuda') if next(self.parameters()).is_cuda else torch.device('cpu')
        out = x
        for m in self.features:
            out = m(out)
            if train:
                bs = out.size(0)
                num_fm = out.size(1)

                fm_count = out.size(1) * out.size(0)
                probs = torch.cuda.FloatTensor(fm_count).uniform_()
                sel_idxs = (probs < self.transf_prob).nonzero()
                out = out.view(out.size(0) * out.size(1), out.size(-2), out.size(-1))
                for idx in sel_idxs:
                    out[idx].data = self.complete_transf(out[idx].detach().cpu()).to(device)
                out = out.view(bs, num_fm, out.size(-2), out.size(-1))


        # out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.ModuleList(layers)


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')
