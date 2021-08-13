from collections import OrderedDict

import timm
import torch
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelMaxPool)


class PoolDict(torch.nn.Module):

    def forward(self, x):
        return torch.cat([
            torch.nn.functional.adaptive_avg_pool2d(v, 1).flatten(1)
            for v in x.values()
        ],
                         dim=1)


class NormalBackboneWithFPN(torch.nn.Module):
    """Backbone with FPN builder adapted from torchvision.models.detection.backbone_utils.BackboneWithFPN
    """

    def __init__(self,
                 backbone_body,
                 in_channels_list,
                 out_channels=256,
                 flatten_fpn=False):
        super().__init__()
        self.body = backbone_body
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.flatten_fpn = flatten_fpn
        if self.flatten_fpn:
            self.pool_dict = PoolDict()

    def forward(self, x):
        x = self.body(x)
        x.device = next(iter(x.values())).device
        x = self.fpn(x)
        x.device = next(iter(x.values())).device
        if self.flatten_fpn:
            x = self.pool_dict(x)
        return x


timm_backbone_body_default_params = {
    'tf_efficientnet_b7_ns': {
        'features_only': True,
        'out_indices': (
            1,
            2,
            3,
            4,
        ),
    },
    'tf_efficientnet_b7_ap': {
        'features_only': True,
        'out_indices': (
            1,
            2,
            3,
            4,
        ),
    },
}


def change_layer(model,
                 source_module,
                 target_module,
                 params_map={},
                 state_dict_map={}):
    """ To change specific type of layers to another one. e.g. BatchNorm2d to FrozenBatchNorm2d
    
    Args:
        model (nn.Module): The model to alter
        source_module (nn.Module Class): the module class to be replaced. It will be used by checking
            if any module in the model is its instance.
        target_module (nn.Module Class): The constructor that will replace all source_module
        params_map (Dict[str: str]): The dict keys are the attributes of the source_module and the
            dict values are the parameter name to construct target_module
        state_dict_map (Dict[str: str]): The dict keys are the state dict key name of the source_module
            and the dict values are the state dict key name of the target_module to be loaded.
    """

    def check_children(model):
        for name, module in model.named_children():
            if isinstance(module, source_module):
                state_dict_to_load = {
                    (k if k not in state_dict_map else state_dict_map[k]): v
                    for k, v in module.state_dict().items()
                }
                setattr(
                    model, name,
                    target_module(
                        **{
                            new_param: getattr(module, org_param)
                            for org_param, new_param in params_map.items()
                        }))
                getattr(model, name).load_state_dict(state_dict_to_load,
                                                     strict=False)
            check_children(getattr(model, name))

    check_children(model)


def freeze_layers(model, stage_name, trainable_stages):
    if trainable_stages >= len(stage_name):
        return
    freeze_to = stage_name[len(stage_name) - trainable_stages - 1]

    if any([
            any([name.startswith(x)
                 for x in freeze_to])
            for name, param in model.named_parameters()
    ]) and freeze_to != '':
        freeze_flag = False
        for name, param in reversed(list(model.named_parameters())):
            if not freeze_flag and any([name.startswith(x) for x in freeze_to]):
                freeze_flag = True
            if freeze_flag:
                param.requires_grad_(False)
        return

    trainable = stage_name[len(stage_name) - trainable_stages:]
    freeze_flag = True
    for name, param in model.named_parameters():
        if freeze_flag and any(
            [name.startswith(y) for x in trainable for y in x]):
            freeze_flag = False
        if freeze_flag:
            param.requires_grad_(False)
    return


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


class TIMMBackboneBodyWrapper(torch.nn.Module):

    def __init__(self, timm_model):
        super().__init__()
        self.timm_model = timm_model

    def forward(self, *args, **kwargs):
        out_list = self.timm_model(*args, **kwargs)
        out_dict = OrderedDict([
            (i if torchvision.__version__ == '0.5.0a0' else str(i), feature)
            for i, feature in enumerate(out_list)
        ])
        return out_dict


def build_timm_backbone_body(backbone_name,
                             pretrained=True,
                             norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                             trainable_stages=3,
                             **kwargs):
    timm_params = timm_backbone_body_default_params[backbone_name]
    timm_params.update(kwargs)
    dummy_timm_backbone_body = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        **{
            k: (True if k == 'features_only' else v)
            for k, v in timm_params.items()
            if 'out_indices' != k
        })
    timm_backbone_body_stage_name = [[
        info['module'].replace('.', '_'), info['module']
    ] for info in dummy_timm_backbone_body.feature_info.info]
    timm_backbone_body = timm.create_model(backbone_name,
                                           pretrained=pretrained,
                                           **timm_params)
    if norm_layer is not torch.nn.BatchNorm2d:
        if torchvision.__version__ > '0.6.1':
            change_layer(timm_backbone_body,
                         torch.nn.BatchNorm2d,
                         norm_layer,
                         params_map={
                             'num_features': 'num_features',
                             'eps': 'eps'
                         })
        else:
            change_layer(timm_backbone_body,
                         torch.nn.BatchNorm2d,
                         norm_layer,
                         params_map={'num_features': 'n'})

    freeze_layers(timm_backbone_body, timm_backbone_body_stage_name,
                  trainable_stages)
    out_channels = timm_backbone_body.feature_info.channels()
    backbone_body = TIMMBackboneBodyWrapper(timm_backbone_body)
    return backbone_body, out_channels


def build_timm_model_not_features(backbone_name,
                                  pretrained=True,
                                  norm_layer=torch.nn.BatchNorm2d,
                                  trainable_stages=100,
                                  **kwargs):
    timm_model = timm.create_model(backbone_name,
                                   pretrained=pretrained,
                                   **kwargs)
    if norm_layer is not torch.nn.BatchNorm2d:
        if torchvision.__version__ > '0.6.1':
            change_layer(timm_model,
                         torch.nn.BatchNorm2d,
                         norm_layer,
                         params_map={
                             'num_features': 'num_features',
                             'eps': 'eps'
                         })
        else:
            change_layer(timm_model,
                         torch.nn.BatchNorm2d,
                         norm_layer,
                         params_map={'num_features': 'n'})

    if hasattr(timm_model, 'feature_info'):
        timm_model_stage_name = [[
            info['module'].replace('.', '_'), info['module']
        ] for info in timm_model.feature_info]
        freeze_layers(timm_model, timm_model_stage_name, trainable_stages)
    else:
        if trainable_stages == 0:
            freeze_all(timm_model)
    return timm_model


def change_conv_one_channels(module,
                             norm_mean_org=None,
                             norm_std_org=None,
                             norm_mean_new=None,
                             norm_std_new=None):
    assert isinstance(module, torch.nn.modules.conv._ConvNd)
    assert module.groups == 1
    if module.in_channels == 1:
        return module

    requires_grad = module.weight.requires_grad
    device = module.weight.device

    new_params = {}
    for key in torch.nn.modules.conv._ConvNd.__constants__:
        if key not in [
                'output_padding', 'in_channels'
        ] and (key in module.__class__.__init__.__code__.co_varnames or
               'kwargs' in module.__class__.__init__.__code__.co_varnames):
            new_params[key] = getattr(module, key)
        elif key == 'in_channels':
            new_params['in_channels'] = 1
    new_params['bias'] = True
    new_module = module.__class__(**new_params)
    if norm_mean_org is not None:
        state_dict = module.state_dict()
        state_dict['bias'] = state_dict.get('bias', 0) + torch.sum(
            state_dict['weight'] * torch.tensor(
                [(norm_mean_new[0] - x) / y
                 for x, y in zip(norm_mean_org, norm_std_org)],
                dtype=state_dict['weight'].dtype,
                device=state_dict['weight'].device)[None, ..., None, None],
            dim=(1, 2, 3))
        state_dict['weight'] = norm_std_new[0] * torch.sum(
            state_dict['weight'] / torch.tensor(
                norm_std_org,
                dtype=state_dict['weight'].dtype,
                device=state_dict['weight'].device)[None, ..., None, None],
            dim=1,
            keepdim=True)
        print('1-channel weight loading: ',
              new_module.load_state_dict(state_dict))
    if not requires_grad:
        for p in new_module.parameters():
            p.requires_grad_(False)
    return new_module.to(device)


def in_channel_modifier(backbone_body, input_nc, norm_mean_org, norm_std_org,
                        norm_mean_new, norm_std_new):
    if input_nc == 3:
        return
    if isinstance(backbone_body,
                  (timm.models.efficientnet.EfficientNet,
                   timm.models.efficientnet.EfficientNetFeatures)):
        if input_nc == 1:
            backbone_body.conv_stem = change_conv_one_channels(
                backbone_body.conv_stem, norm_mean_org, norm_std_org,
                norm_mean_new, norm_std_new)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def timm_fpn_backbone(backbone_name,
                      pretrained,
                      input_nc=3,
                      weights=None,
                      unfix_backbone_body=False,
                      norm_layer='default',
                      features_only=True,
                      flatten_fpn=False,
                      org_norm_mean=None,
                      org_norm_std=None,
                      new_norm_mean=None,
                      new_norm_std=None,
                      **kwargs):
    if norm_layer == 'default':
        if features_only:
            norm_layer = misc_nn_ops.FrozenBatchNorm2d
        else:
            norm_layer = torch.nn.BatchNorm2d
    #assert input_nc == 3 or not pretrained
    if not features_only:
        backbone_body = build_timm_model_not_features(
            backbone_name,
            pretrained=pretrained,
            norm_layer=norm_layer,
            trainable_stages=(100 if unfix_backbone_body else 3),
            **kwargs)
    else:
        backbone_body, out_channels_list = build_timm_backbone_body(
            backbone_name,
            pretrained,
            norm_layer=norm_layer,
            trainable_stages=(100 if unfix_backbone_body else 3),
            **kwargs)
        assert len(out_channels_list) == 4

    in_channel_modifier(
        backbone_body if not features_only else backbone_body.timm_model,
        input_nc, org_norm_mean, org_norm_std, new_norm_mean, new_norm_std)

    if weights is not None:
        if any([x.startswith('timm_model.') for x in weights]) and not any(
            [x.startswith('timm_model.') for x in backbone_body.state_dict()]):
            weights = {(k[len('timm_model.'):] if 'timm_model.' in k else k): v
                       for k, v in weights.items()}
        elif not any([x.startswith('timm_model.') for x in weights]) and all(
            [x.startswith('timm_model.') for x in backbone_body.state_dict()]):
            weights = {('timm_model.' + k): v for k, v in weights.items()}
        msg = backbone_body.load_state_dict(weights, strict=False)
        print('Load backbone pretraining:', msg)

    if not features_only:
        model = backbone_body
    else:
        model = NormalBackboneWithFPN(backbone_body,
                                      out_channels_list,
                                      flatten_fpn=flatten_fpn)

    return model
