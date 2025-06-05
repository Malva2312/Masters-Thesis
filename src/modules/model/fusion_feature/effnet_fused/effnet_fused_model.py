import torch
from src.modules.model.standalone.efficientnet.efficientnet_model import EfficientNetModel

import torch.nn as nn

class EffNet_Fused_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.effnet_model = EfficientNetModel(config)

        # Parse extractors from config
        self.extractors = config.effnet_config.get('extractors', [])
        self.default_layer = config.effnet_config.get('default_layer', 'blocks.5')

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
            if aux.ndim != 3:
                raise ValueError(f"Expected aux input '{name}' to have shape (B, H, W), got {aux.shape}")

            aux = aux.to(x.device)
            B, H, W = aux.shape

            if H > 1 and W > 1:
                aux = aux.unsqueeze(1)
                if name not in self.projectors:
                    self.projectors[name] = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    ).to(x.device)
                proj = self.projectors[name](aux)
            else:
                aux = aux.view(B, -1)
                if name not in self.projectors:
                    self.projectors[name] = nn.Sequential(
                        nn.Linear(aux.shape[1], 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU()
                    ).to(x.device)
                proj = self.projectors[name](aux)
                proj = proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])

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

        if len(proj_list) > 1:
            target_h, target_w = x.shape[2], x.shape[3]
            proj_list_resized = []
            for p in proj_list:
                if p.shape[2] != target_h or p.shape[3] != target_w:
                    p = torch.nn.functional.interpolate(p, size=(target_h, target_w), mode='nearest')
                proj_list_resized.append(p)
            fusion_key = f"{layer_name}_multi"
            total_channels = sum(p.shape[1] for p in proj_list_resized)
            if fusion_key not in self.fusion_convs:
                self.fusion_convs[fusion_key] = nn.Conv2d(
                    in_channels=total_channels,
                    out_channels=x.shape[1],
                    kernel_size=1
                ).to(x.device)
            x = self.fusion_convs[fusion_key](torch.cat(proj_list_resized, dim=1))
        else:
            x = proj_list[0]

        return x