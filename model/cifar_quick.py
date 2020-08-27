# quick model with 3 or 4-conv layers for CIFAR
from __future__ import division
import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import cpu
import mxnet.autograd as ag
import mxnet as mx

class QuickBlock(HybridBlock):
    def __init__(self, channels, kernel_size, strides, padding, pooling='avgPool', prefix=''):
        super(QuickBlock, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential(prefix)
            self.block.add(nn.Conv2D(channels=channels, kernel_size=kernel_size,
                                     strides=strides, padding=padding))
            if pooling=='avgPool':
                self.block.add(nn.Activation('relu'))
                self.block.add(nn.AvgPool2D(pool_size=3, strides=2))
            elif pooling=='maxPool':
                self.block.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.block.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.block(x)


class CIFARQuick(HybridBlock):

    def __init__(self, block, fix_layers, pooling, channels, classes, fix_conv=False, **kwargs):
        super(CIFARQuick, self).__init__()
        self.fix_conv = fix_conv
        self.fix_layers = fix_layers
        assert 'fw' in kwargs.keys(), 'no_fw'
        self.fw = kwargs['fw']
        assert 'fix_fc' in kwargs.keys(), 'no_fix_fc'
        self.fix_fc =kwargs['fix_fc']
        with self.name_scope():
            self.feats1 = nn.HybridSequential()
            self.feats1.add(block(channels[0], 5, 1, 2,pooling[0]))
            self.feats2 = nn.HybridSequential()
            self.feats2.add(block(channels[1], 5, 1, 2,pooling[1]))
            self.feats3 = nn.HybridSequential()
            self.feats3.add(block(channels[2], 5, 1, 2,pooling[2]))

            self.feats3.add(nn.Flatten())

            self.fc1 = nn.Dense(64, use_bias=False)
            self.fc2 = nn.Dense(classes, use_bias=False)
            self.fc3 = nn.Dense(5,use_bias=False)
            self.fc4 = nn.Dense(5,use_bias=False)
            self.fc5 = nn.Dense(5, use_bias=False)
            self.fc6 = nn.Dense(5, use_bias=False)
            self.fc7 = nn.Dense(5, use_bias=False)
            self.fc8 = nn.Dense(5, use_bias=False)
            self.fc9 = nn.Dense(5, use_bias=False)
            self.fc10 = nn.Dense(5, use_bias=False)

    def hybrid_forward(self, F, x, num=0, fix_conv=False):
        if self.fix_layers == 0:
            out = F.L2Normalization(self.fc1(self.feats3(self.feats2(self.feats1(x)))))
        elif self.fix_layers == 1:
            with ag.pause():
                x = self.feats1(x)
            out = F.L2Normalization(self.fc1(self.feats3(self.feats2(x))))
        elif self.fix_layers == 2:
            with ag.pause():
                x = self.feats2(self.feats1(x))
            out = F.L2Normalization(self.fc1(self.feats3(x)))
        elif self.fix_layers == 3:
            if self.fix_fc:
                with ag.pause():
                    x = self.fc1(self.feats3(self.feats2(self.feats1(x))))
                out = F.L2Normalization(x)
            else:
                with ag.pause():
                    x = self.feats3(self.feats2(self.feats1(x)))
                out = F.L2Normalization(self.fc1(x))

        if self.fw:
            for i in range(num + 1):
                if i < num:
                    with ag.pause():
                        fc = eval('self.fc' + str(i + 2))
                        if i == 0:
                            output = fc(out)
                        else:
                            output = mx.nd.concat(output, fc(out), dim=1)

                else:
                    fc = eval('self.fc' + str(i + 2))
                    if i == 0:
                        output = fc(out)
                    else:
                        output = mx.nd.concat(output, fc(out), dim=1)
            return out, output

        else:
            for i in range(num + 1):
                fc = eval('self.fc' + str(i + 2))
                if i == 0:
                    output = fc(out)
                else:
                    output = mx.nd.concat(output, fc(out), dim=1)
            return out, output


def quick_cnn(classes, fix_layers,**kwargs):
    assert fix_layers<4
    assert 'fix_fc' in kwargs.keys(), 'no_fix-fc'
    fix_fc = kwargs['fix_fc']
    if fix_fc:
        assert fix_layers==3
    if fix_layers==0:
        fix_conv=False
    else:
        fix_conv=True

    channels = [32,32,64]
    pooling = ['maxPool','avgPool','avgPool']

    net = CIFARQuick(QuickBlock, fix_layers, pooling, channels, classes, fix_conv,**kwargs)

    return net
