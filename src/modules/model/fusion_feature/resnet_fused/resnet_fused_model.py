import torch
import torch.nn as nn
from src.modules.model.standalone.resnet.resnet_model import ResNetModel

class ResNet_Fused_Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet_model = ResNetModel(config)

        # Parse extractors from config
        self.extractors = config.resnet_config.get('extractors', [])
        self.default_layer = config.resnet_config.get('default_layer', 'layer3')

        # Track which extractor injects at which layer
        self.layer_map = {}  # e.g., {"layer3": ["lbp", "rad"]}

        for extractor in self.extractors:
            name = extractor['name']
            layer = extractor.get('layer', self.default_layer)
            self.layer_map.setdefault(layer, []).append(name)

        # Projectors initialized on first forward call
        self.projectors = nn.ModuleDict()
        self.fusion_convs = nn.ModuleDict()  # Optional: handle channel mismatches


    def forward(self, data):
        model_input = data['image']  # (B, C, H, W)
        # If input is grayscale, repeat channels to get 3 channels
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)

        # Forward through standard ResNet layers
        # Model All Layers: odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
        x = self.resnet_model.model.conv1(model_input)
        x = self.resnet_model.model.bn1(x)
        x = self.resnet_model.model.relu(x)
        x = self.resnet_model.model.maxpool(x)

        x = self._fuse_layer('layer1', self.resnet_model.model.layer1(x), data)
        x = self._fuse_layer('layer2', self.resnet_model.model.layer2(x), data)
        x = self._fuse_layer('layer3', self.resnet_model.model.layer3(x), data)
        x = self._fuse_layer('layer4', self.resnet_model.model.layer4(x), data)
       
        x = self.resnet_model.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet_model.model.fc(x)

        return x

    def _fuse_layer(self, layer_name, x, aux_input):
        if aux_input is None or layer_name not in self.layer_map:
            return x

        proj_list = [x]  # Start with the main ResNet feature map

        for name in self.layer_map[layer_name]: # for each extractor vector (B, 1, N) -> B(B, 1, H * W) -> (B, 1, H, W) -> (B, C, H, W)
            if name not in aux_input:
                continue

            aux = aux_input[name]  # Expected shape: (B, 1, N) # N is the extractor feature dimension
            if aux.ndim != 3 or aux.shape[1] != 1:
                raise ValueError(f"Expected aux input '{name}' to have shape (B, 1, N), got {aux.shape}")

            aux = aux.to(x.device)
            B, _, N = aux.shape

            if name not in self.projectors:
                self.projectors[name] = nn.Sequential(
                        nn.Linear(aux.shape[-1], x.shape[-1] * x.shape[-2]),
                        nn.BatchNorm1d( x.shape[-1] * x.shape[-2]),
                        nn.ReLU()
                ).to(x.device)
            proj = self.projectors[name](aux)  # (B, 1, H * W) # H * W == Net Layer Dim
            proj = proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,  x.shape[-2],  x.shape[-1])  # (B, 1, H, W)

            # Reshape # Match ResNet branch channels if needed
            if proj.shape[1] != x.shape[1]:
                fusion_key = f"{layer_name}_{name}"
                if fusion_key not in self.fusion_convs:
                    self.fusion_convs[fusion_key] = nn.Conv2d(
                        in_channels=proj.shape[1],
                        out_channels=x.shape[1],
                        kernel_size=1
                    ).to(x.device)
                proj = self.fusion_convs[fusion_key](proj) # (B, C, H, W)

            proj_list.append(proj)

        # Multi-branch fusion (ResNet + all aux branches)
        if len(proj_list) > 1:
            x = proj_list[0]
            for proj in proj_list[1:]:
                x = x + proj # add alpha fusion as treinable parameter
        else:
            x = proj_list[0]

        return x
