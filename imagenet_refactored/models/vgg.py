""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [ 'vgg11_bn']

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class vgg_rep(nn.Module):
    def __init__(self, blocks):
        super(vgg_rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n

        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        return out

class vgg_greedy(nn.Module):
    def __init__(self, config_vgg, in_size=224,**kwargs):
        super(vgg_greedy, self).__init__()
        self.make_layers(config_vgg, in_size=in_size, **kwargs)
        self.blocks = nn.ModuleList(self.blocks)
        self.main_cnn = vgg_rep(self.blocks)

        self.auxillary_nets[len(self.auxillary_nets)-1] = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            View( 512*7*7),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
        )
        self._initialize_weights()

    def make_layers(self, cfg, in_size, **kwargs):
        self.blocks = []
        self.auxillary_nets = nn.ModuleList([])
        in_channels = 3
        avg_size = 112
        last_M = False
        for v in cfg:
            if v == 'M':
                layer = [nn.MaxPool2d(kernel_size=2, stride=2)]
                avg_size = int(avg_size / 2)
                in_size = int(in_size / 2)
                last_M = True
                continue
            else:
                if last_M:
                    last_M=False
                else:
                    layer = []
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            in_channels = v
            self.blocks.append(nn.Sequential(*layer))


            model_c = auxillary_classifier(in_size=in_size,
                                           n_lin=kwargs['nlin'], 
                                           input_features=in_channels,
                                           num_classes=1000).cuda()
            self.auxillary_nets.append(model_c)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0)

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation



class auxillary_classifier(nn.Module):
    def __init__(self,
                 input_features=256, in_size=32,
                 num_classes=1000,n_lin=2):
        super(auxillary_classifier, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size

        feature_size = 2*input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            bn_temp = nn.BatchNorm2d(feature_size)


            conv = nn.Conv2d(input_features, feature_size,
                             kernel_size=3, stride=1, padding=0, bias=False)
            self.blocks.append(nn.Sequential(conv,bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        self.bn = nn.BatchNorm2d(feature_size)

        self.classifier = nn.Linear(feature_size*4, num_classes)


    def forward(self, x):
        out = x
        #First reduce the size by x4
        out = F.adaptive_avg_pool2d(out,(math.ceil(self.in_size/2),math.ceil(self.in_size/2)))

        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, (2,2))
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg11_bn(**kwargs):
    model = vgg_greedy(cfg['A'], **kwargs)
    return model


