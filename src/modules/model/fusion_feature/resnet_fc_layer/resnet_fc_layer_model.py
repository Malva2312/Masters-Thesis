import torch
from src.modules.model.standalone.resnet.resnet_model import ResNetModel

class ResNet_FC_Layer_Model(torch.nn.Module):
    def __init__(self, config, fusion_feature_dim=None):
        super().__init__()
        self.config = config
        self.device_name = getattr(config, "device", "cpu")
        self.fusion_layer = config.get("fusion_layer", "layer3")

        self.resnet = ResNetModel(config=config)
        self.fusion_feature_dim = fusion_feature_dim if fusion_feature_dim is not None else config.get("fusion_feature_dim", 512)
        fusion_dim = self._get_layer_output_dim(self.fusion_layer)

        # Fusion logic
        self.aux_fc = torch.nn.Linear(self.fusion_feature_dim, fusion_dim)
        self.fusion_proj_conv = torch.nn.Conv2d(
            in_channels=fusion_dim * 2,
            out_channels=fusion_dim,
            kernel_size=1
        )

        self.to(torch.device(self.device_name))

    def update_fusion_dim(self, new_fusion_dim):
        self.fusion_feature_dim = new_fusion_dim
        fusion_dim = self._get_layer_output_dim(self.fusion_layer)
        self.aux_fc = torch.nn.Linear(new_fusion_dim, fusion_dim)
        self.aux_fc.to(next(self.parameters()).device)
        self.fusion_proj_conv = torch.nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1)
        self.fusion_proj_conv.to(next(self.parameters()).device)

    def _get_layer_output_dim(self, layer_name):
        dummy_input = torch.randn(1, 3, 224, 224).to(next(self.resnet.parameters()).device)
        x = dummy_input
        for name, layer in self.resnet.model.named_children():
            x = layer(x)
            if name == layer_name:
                return x.shape[1]
        raise ValueError(f"Invalid layer name for fusion: {layer_name}")

    def _inject_fusion(self, x, aux_input):
        fused = False
        for name, layer in self.resnet.model.named_children():
            if name == "fc":
                continue  # skip final FC here, we'll apply it manually later

            x = layer(x)

            if not fused and name == self.fusion_layer:
                if aux_input is None:
                    raise ValueError("aux_input must be provided for fusion at layer '{}'".format(self.fusion_layer))
                aux_proj = self.aux_fc(aux_input).unsqueeze(-1).unsqueeze(-1)
                aux_proj = aux_proj.expand(-1, -1, x.shape[2], x.shape[3])
                x = torch.cat([x, aux_proj], dim=1)
                x = self.fusion_proj_conv(x)  # Project back to original channel size
                fused = True

        return x

    def forward(self, model_input, aux_input=None):
        # Ensure 3-channel input
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)

        x = self._inject_fusion(model_input, aux_input)

        # Apply global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten to (B, C)

        # Final classification layer
        x = self.resnet.model.fc(x)

        return x
