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
# pylint: disable= arguments-differ,unused-argument,missing-docstring,too-many-lines
"""ResNets, implemented in Gluon."""
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
import mxnet as mx
import mxnet.autograd as ag

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class BasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

# Nets
class ResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__()
        assert len(layers) == len(channels) - 1
        assert 'fw' in kwargs.keys(), 'no_fw'
        self.fw = kwargs['fw']
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.GlobalAvgPool2D())

            c_way = 10
            # self.output = nn.Dense(classes, in_units=channels[-1])
            self.fc0 = nn.Dense(classes, in_units=channels[-1], use_bias=False)
            self.fc1 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc2 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc3 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc4 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc5 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc6 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc7 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc8 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc9 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)
            self.fc10 = nn.Dense(c_way, in_units=channels[-1], use_bias=False)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x, num=0, fix_cnn=False):
        # x = self.features(x)
        # x = self.output(x)
        if fix_cnn:
            with ag.pause():
                x = self.features[:7](x)
                x = self.features[7][0](x)
            x = self.features[7][1](x)
            x = self.features[8:](x)
            feat = F.L2Normalization(x)
            out = feat
        else:
            x = self.features(x)
            feat = F.L2Normalization(x)
            out = feat


        if self.fw:

            for i in range(num+1):
                if i < num:
                    with ag.pause():
                        fc = eval('self.fc' + str(i))
                        if i == 0:
                            output = fc(out)
                        else:
                            output = mx.nd.concat(output, fc(out), dim=1)
                else:
                    fc = eval('self.fc' + str(i))
                    if i == 0:
                        output = fc(out)
                    else:
                        output = mx.nd.concat(output, fc(out), dim=1)
            return feat, output

        else:
            for i in range(num+1):
                fc = eval('self.fc'+str(i))
                if i == 0:
                    output = fc(out)
                else:
                    output = mx.nd.concat(output, fc(out), dim=1)
            return feat, output

from gluoncv.model_zoo import resnet18_v1

# Constructor
def get_resnet(wo_bn, **kwargs):
    layers, channels = [2, 2, 2, 2], [64, 64, 128, 256, 512]
    resnet_class = ResNetV1
    block_class = BasicBlockV1
    if wo_bn:
        net = resnet_class(block_class, layers, channels, norm_kwargs={'use_global_stats': True}, **kwargs)
    else:
        net = resnet_class(block_class, layers, channels, **kwargs)
    return net


def resnet18_v1(wo_bn, **kwargs):
    return get_resnet(wo_bn,**kwargs)
