import torch
from src.modules.model.standalone.effnet.efficient_net_model import EfficientNetModel

import torch.nn as nn

class EffNet_Fused_Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.effnet_model = EfficientNetModel(config)

        # Parse extractors from config
        self.extractors = config.effnet_config.get('extractors', [])
        # Set default_layer to the last block if not provided
        num_blocks = len(self.effnet_model.model.features) - 1  # exclude features[0]
        self.default_layer = config.effnet_config.get('default_layer', f'blocks.{num_blocks - 1}')

        # Track which extractor injects at which layer
        self.layer_map = {}  # e.g., {"blocks.5": ["lbp", "rad"]}

        for extractor in self.extractors:
            name = extractor['name']
            layer = extractor.get('layer', self.default_layer)
            self.layer_map.setdefault(layer, []).append(name)

        # Projectors initialized on first forward call
        self.projectors = nn.ModuleDict()
        self.fusion_convs = nn.ModuleDict()

    def forward(self, data):
        model_input = data['image']
        # If input is grayscale, repeat channels to get 3 channels
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)

        x = self.effnet_model.model.features[0](model_input)
        for i, block in enumerate(self.effnet_model.model.features[1:]):
            layer_name = f'blocks.{i}'
            x = block(x)
            x = self._fuse_layer(layer_name, x, data)

        x = self.effnet_model.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.effnet_model.model.classifier(x)
        return x

    def _fuse_layer(self, layer_name, x, aux_input):
        if aux_input is None or layer_name not in self.layer_map:
            return x

        proj_list = [x]

        for name in self.layer_map[layer_name]:
            if name not in aux_input:
                continue
            aux = aux_input[name]

            aux = aux.to(x.device)
            aux = aux.view(aux_input['image'].shape[0], aux_input['image'].shape[1], 1, -1)
            B, C, _, N = aux.shape

            if name not in self.projectors:
                self.projectors[name] = nn.Sequential(
                        nn.Linear(aux.shape[-1], x.shape[-1] * x.shape[-2]),
                        nn.BatchNorm1d(x.shape[-1] * x.shape[-2]),
                        nn.ReLU()
                ).to(x.device)
            
            # Process each channel separately, then stack
            proj_channels = []
            for c in range(C):
                aux_c = aux[:, c, :, :].reshape(B, -1)  # (B, H*W)
                proj_c = self.projectors[name](aux_c)  # (B, H*W)
                proj_c = proj_c.view(B, 1, x.shape[-2], x.shape[-1])  # (B, 1, H, W)
                proj_channels.append(proj_c)
            proj = torch.cat(proj_channels, dim=1)  # (B, C, H, W)
        
            # If the projected shape does not match x, apply a 1x1 conv to adjust channels
            if proj.shape[1] != x.shape[1]:
                fusion_key = f"{layer_name}_{name}"
                if fusion_key not in self.fusion_convs:
                    self.fusion_convs[fusion_key] = nn.Conv2d(
                        in_channels=proj.shape[1],
                        out_channels=x.shape[1],
                        kernel_size=1
                    ).to(x.device)
                proj = self.fusion_convs[fusion_key](proj)

            proj_list.append(proj)

        # Multi-branch fusion (EffNet + all aux branches)
        if len(proj_list) > 1:
            for i, name in enumerate(self.layer_map[layer_name]):
                if i + 1 < len(proj_list):  # proj_list[0] is main, proj_list[1:] are aux
                    x = x + proj_list[i + 1]
        else:
            x = proj_list[0]

        return x