import torch
from src.modules.model.efficient_net.efficient_net_model import EfficientNetModel
from src.modules.features.lbp_extractor import LocalBinaryPattern

class EffNet_LBP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.efficient_net = EfficientNetModel(config=config)
        self.lbp_extractor = LocalBinaryPattern()
        # Assume LBP histogram length is 59 for 'uniform' with P=8, R=1
        lbp_feature_dim = 59
        
        effnet_out_dim = self.efficient_net.model.classifier[1].in_features
        self.fc_lbp = torch.nn.Linear(lbp_feature_dim, 128)
        self.classifier = torch.nn.Linear(effnet_out_dim + 128, config.number_of_classes)

    def forward(self, image):
        # EfficientNet features
        x = image.to(next(self.parameters()).device)
        effnet_features = self.efficient_net.model.features(x.repeat(1, 3, 1, 1))
        effnet_features = self.efficient_net.model.avgpool(effnet_features)
        effnet_features = torch.flatten(effnet_features, 1)
        # LBP features
        with torch.no_grad():
            lbp_imgs = self.lbp_extractor(image)
        # Compute LBP histogram for each image in batch
        lbp_hist = []
        for lbp_img in lbp_imgs:
            hist = torch.histc(lbp_img.float(), bins=59, min=0, max=58)
            hist = hist / (hist.sum() + 1e-6)
            lbp_hist.append(hist)
        lbp_hist = torch.stack(lbp_hist).to(x.device)
        lbp_features = self.fc_lbp(lbp_hist)
        # Concatenate and classify
        features = torch.cat([effnet_features, lbp_features], dim=1)
        out = self.classifier(features)
        return out
