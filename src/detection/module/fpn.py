from collections import OrderedDict

import torch
import torchvision
from torch import nn


class MSCAM(nn.Module):
    """Module needed in AttentionalFeatureFusionFPN"""

    def __init__(self, num_channels, r):
        super().__init__()
        bottleneck = num_channels // r
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.w1 = nn.Conv2d(num_channels, bottleneck, 1)
        self.w2 = nn.Conv2d(bottleneck, num_channels, 1)
        self.pwc1 = nn.Conv2d(num_channels, bottleneck, 1)
        self.pwc2 = nn.Conv2d(bottleneck, num_channels, 1)
        self.gn_w1 = nn.GroupNorm(num_groups=32, num_channels=bottleneck)
        self.gn_w2 = nn.GroupNorm(num_groups=32, num_channels=num_channels)
        self.gn_pwc1 = nn.GroupNorm(num_groups=32, num_channels=bottleneck)
        self.gn_pwc2 = nn.GroupNorm(num_groups=32, num_channels=num_channels)

    def forward(self, x):
        x1 = self.pool(x)
        x1 = self.w1(x1)
        x1 = self.gn_w1(x1).relu()
        x1 = self.w2(x1)
        x1 = self.gn_w2(x1)

        x2 = self.pwc1(x)
        x2 = self.gn_pwc1(x2).relu()
        x2 = self.pwc2(x2)
        x2 = self.gn_pwc2(x2)

        return (x1 + x2).sigmoid()


class AttentionalFeatureFusionFPN(nn.Module):
    """ Deprecated, please use ModulerFPN with iAFF=True. Re-implementation of the paper: "Attentional Feature Fusion" """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels,
                                           out_channels,
                                           3,
                                           padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(
                extra_blocks,
                torchvision.ops.feature_pyramid_network.ExtraFPNBlock)
        self.extra_blocks = extra_blocks

        self.MSCAM1s = nn.ModuleList(
            [MSCAM(out_channels, 4) for _ in range(len(in_channels_list))])
        self.MSCAM2s = nn.ModuleList(
            [MSCAM(out_channels, 4) for _ in range(len(in_channels_list))])

    @staticmethod
    def iAFF(MSCAM1, MSCAM2, x, y):
        assert x.shape == y.shape, f"input shape is not the same: {x.shape}, {y.shape}"
        att_weight = MSCAM2(MSCAM1(x + y))
        return att_weight * x + (1 - att_weight) * y

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for i, (feature, inner_block, layer_block) in enumerate(
                zip(
                    x[:-1][::-1],
                    self.inner_blocks[:-1][::-1],
                    self.layer_blocks[:-1][::-1],
                )):
            if not inner_block:
                continue
            inner_lateral = inner_block(feature)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = nn.functional.interpolate(last_inner,
                                                       size=feat_shape,
                                                       mode="nearest")
            last_inner = self.iAFF(self.MSCAM1s[i], self.MSCAM2s[i],
                                   inner_lateral, inner_top_down)
            results.insert(0, layer_block(last_inner))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class AttentionGuidedFPN(nn.Module):
    """ Deprecated, please use ModulerFPN with ACFPN=True. Re-implementation of the CEM module in the paper:
    "Attention-guided Context Feature Pyramid Network for Object Detection".
    Theis implementation is based on this repository:
    https://github.com/Caojunxu/AC-FPN
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels,
                                           out_channels,
                                           3,
                                           padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(
                extra_blocks,
                torchvision.ops.feature_pyramid_network.ExtraFPNBlock)
        self.extra_blocks = extra_blocks
        self.num_dilations = [3, 6, 12, 18, 24]
        aspp_blocks = []
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 256
        dim_in = in_channels_list[-1]
        for i, dilation in enumerate(self.num_dilations):
            aspp_blocks.append(
                self.dense_aspp_block(input_num=dim_in + d_feature1 * i,
                                      num1=d_feature0,
                                      num2=d_feature1,
                                      dilation_rate=dilation,
                                      drop_out=dropout0))
        self.aspp_blocks = torch.nn.ModuleList(aspp_blocks)
        self.CEM_final_conv = torch.nn.Conv2d(
            len(self.num_dilations) * d_feature1, out_channels, 1)
        self.CEM_final_gn = torch.nn.GroupNorm(num_groups=32,
                                               num_channels=out_channels)

    @staticmethod
    def dense_aspp_block(input_num, num1, num2, dilation_rate, drop_out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_num, num1, kernel_size=1),
            torch.nn.GroupNorm(num_groups=32, num_channels=num1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num1,
                            num2,
                            kernel_size=3,
                            padding=dilation_rate,
                            dilation=dilation_rate), torch.nn.ReLU(),
            torch.nn.Dropout(drop_out))

    @staticmethod
    def iAFF(MSCAM1, MSCAM2, x, y):
        assert x.shape == y.shape, f"input shape is not the same: {x.shape}, {y.shape}"
        att_weight = MSCAM2(MSCAM1(x + y))
        return att_weight * x + (1 - att_weight) * y

    def dense_aspp_forward(self, _input):
        conv_outs = []

        conv_out = self.aspp_blocks[0](_input)
        if 0 != len(self.num_dilations) - 1:
            x = torch.cat((conv_out, _input), dim=1)
            conv_outs.append(conv_out)

        for i, dilation in enumerate(self.num_dilations[1:], 1):
            conv_out = self.aspp_blocks[i](x)
            if i != len(self.num_dilations) - 1:
                x = torch.cat((conv_out, x), dim=1)
            conv_outs.append(conv_out)
        x = torch.cat(conv_outs, dim=1)
        x = self.CEM_final_conv(x)
        x = self.CEM_final_gn(x)
        return x

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.inner_blocks[-1](x[-1]) + self.dense_aspp_forward(
            x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_lateral = inner_block(feature)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(last_inner,
                                                             size=feat_shape,
                                                             mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class ModulerFPN(nn.Module):
    """
    Combination of ACFPN and iAFF
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.ACFPN = self.iAFF = False
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in self.in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels,
                                           out_channels,
                                           3,
                                           padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(
                extra_blocks,
                torchvision.ops.feature_pyramid_network.ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def setup_ACFPN(self):
        self.ACFPN = True
        self.num_dilations = [3, 6, 12, 18, 24]
        aspp_blocks = []
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 256
        dim_in = self.in_channels_list[-1]
        for i, dilation in enumerate(self.num_dilations):
            aspp_blocks.append(
                self.dense_aspp_block(input_num=dim_in + d_feature1 * i,
                                      num1=d_feature0,
                                      num2=d_feature1,
                                      dilation_rate=dilation,
                                      drop_out=dropout0))
        self.aspp_blocks = torch.nn.ModuleList(aspp_blocks)
        self.CEM_final_conv = torch.nn.Conv2d(
            len(self.num_dilations) * d_feature1, self.out_channels, 1)
        self.CEM_final_gn = torch.nn.GroupNorm(num_groups=32,
                                               num_channels=self.out_channels)

    def setup_iAFF(self):
        self.iAFF = True
        self.MSCAM1s = nn.ModuleList([
            MSCAM(self.out_channels, 4)
            for _ in range(len(self.in_channels_list))
        ])
        self.MSCAM2s = nn.ModuleList([
            MSCAM(self.out_channels, 4)
            for _ in range(len(self.in_channels_list))
        ])

    @staticmethod
    def apply_iAFF(MSCAM1, MSCAM2, x, y):
        assert x.shape == y.shape, f"input shape is not the same: {x.shape}, {y.shape}"
        att_weight = MSCAM2(MSCAM1(x + y))
        return att_weight * x + (1 - att_weight) * y

    @staticmethod
    def dense_aspp_block(input_num, num1, num2, dilation_rate, drop_out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_num, num1, kernel_size=1),
            torch.nn.GroupNorm(num_groups=32, num_channels=num1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num1,
                            num2,
                            kernel_size=3,
                            padding=dilation_rate,
                            dilation=dilation_rate), torch.nn.ReLU(),
            torch.nn.Dropout(drop_out))

    def dense_aspp_forward(self, _input):
        conv_outs = []

        conv_out = self.aspp_blocks[0](_input)
        if 0 != len(self.num_dilations) - 1:
            x = torch.cat((conv_out, _input), dim=1)
            conv_outs.append(conv_out)

        for i, dilation in enumerate(self.num_dilations[1:], 1):
            conv_out = self.aspp_blocks[i](x)
            if i != len(self.num_dilations) - 1:
                x = torch.cat((conv_out, x), dim=1)
            conv_outs.append(conv_out)
        x = torch.cat(conv_outs, dim=1)
        x = self.CEM_final_conv(x)
        x = self.CEM_final_gn(x)
        return x

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.inner_blocks[-1](x[-1])

        if self.ACFPN:
            last_inner += self.dense_aspp_forward(x[-1])

        results = []
        results.append(self.layer_blocks[-1](last_inner))
        for i, (feature, inner_block, layer_block) in enumerate(
                zip(x[:-1][::-1], self.inner_blocks[:-1][::-1],
                    self.layer_blocks[:-1][::-1])):
            if not inner_block:
                continue
            inner_lateral = inner_block(feature)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(last_inner,
                                                             size=feat_shape,
                                                             mode="nearest")

            if not self.iAFF:
                last_inner = inner_lateral + inner_top_down
            else:
                last_inner = self.apply_iAFF(self.MSCAM1s[i], self.MSCAM2s[i],
                                             inner_lateral, inner_top_down)

            results.insert(0, layer_block(last_inner))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
