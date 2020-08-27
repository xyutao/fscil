# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument
"""ResNets, implemented in Gluon."""
from __future__ import division

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
import mxnet as mx
from mxnet import autograd as ag

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [16, 16, 32, 64]
    layers = [n] * (len(channels) - 1)
    return layers, channels

# Blocks
class CIFARBasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(CIFARBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

# Nets
class CIFARResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are CIFARBasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 10
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, block, layers, channels, classes=10,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):

        super(CIFARResNetV1, self).__init__()
        assert len(layers) == len(channels) - 1
        assert 'fw' in kwargs.keys(), 'no_fw'
        self.fw=kwargs['fw']
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())

            # self.output = nn.Dense(classes, in_units=channels[-1])
            self.fc2 = nn.Dense(classes, in_units=channels[-1], use_bias=False)
            self.fc3 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc4 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc5 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc6 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc7 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc8 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc9 = nn.Dense(5, in_units=channels[-1], use_bias=False)
            self.fc10 = nn.Dense(5, in_units=channels[-1], use_bias=False)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x, num=0, fix_cnn=False):
        if fix_cnn:
            with ag.pause():
                x = self.features[:4](x)
                x = self.features[4][0:2](x)
            x=self.features[4][2](x)
            x=self.features[5:](x)
            out = F.L2Normalization(x)
            feat = out
        else:
            x = self.features(x)
            out = F.L2Normalization(x)
            feat = out
        if self.fw:

            for i in range(num+1):
                if i < num:
                    with ag.pause():
                        fc = eval('self.fc' + str(i+2))
                        if i == 0:
                            output = fc(out)
                        else:
                            output = mx.nd.concat(output, fc(out), dim=1)
                else:
                    fc = eval('self.fc' + str(i+2))
                    if i == 0:
                        output = fc(out)
                    else:
                        output = mx.nd.concat(output, fc(out), dim=1)
            return feat, output

        else:

            for i in range(num+1):
                fc = eval('self.fc'+str(i+2))
                if i == 0:
                    output = fc(out)
                else:
                    output = mx.nd.concat(output, fc(out), dim=1)
            return feat, output



# Constructor
def get_cifar_resnet(num_layers,**kwargs):

    layers, channels = _get_resnet_spec(num_layers)
    resnet_class = CIFARResNetV1
    block_class = CIFARBasicBlockV1
    assert 'wo_bn' in kwargs.keys(),'no_bn'
    bn = kwargs['wo_bn']
    if bn:
       net = resnet_class(block_class, layers, channels, norm_kwargs={'use_global_stats': True}, **kwargs)
    else:
       net = resnet_class(block_class, layers, channels, **kwargs)
    return net


def cifar_resnet20_v1(**kwargs):

    return get_cifar_resnet(20, **kwargs)

