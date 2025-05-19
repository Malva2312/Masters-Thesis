import torch

from src.modules.model.efficient_net.efficient_net_model import EfficientNetModel
from src.modules.model.linear_svm.linear_svm_model import LinearSVMModel

from src.modules.features.lbp_extractor import LocalBinaryPattern


class EffNetSVMFusionModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.effnet = EfficientNetModel(config=config.effnet_config)
        self.svm = LinearSVMModel(input_dim=config.svm_config.input_dim)
        self.lbp_extractor = LocalBinaryPattern(
            P=getattr(config, "lbp_P", 8),
            R=getattr(config, "lbp_R", 1),
            method=getattr(config, "lbp_method", "uniform")
        )
        self.device_name = config.device

    def forward(self, image):
        # EfficientNet
        effnet_logits = self.effnet(image.to(self.device_name))
        effnet_probs = torch.sigmoid(effnet_logits[:, 0])
        effnet_pred = (effnet_probs > 0.5).long().unsqueeze(1)

        # SVM
        lbp = self.lbp_extractor(image)
        svm_input = lbp.view(lbp.size(0), -1).to(self.device_name)
        svm_logits = self.svm(svm_input)
        svm_pred = ((torch.sign(svm_logits) + 1) // 2).long()

        # Late fusion
        fused_pred = ((effnet_pred + svm_pred) >= 1).long()

        return {
            "effnet_logits": effnet_logits,
            "effnet_pred": effnet_pred,
            "svm_logits": svm_logits,
            "svm_pred": svm_pred,
            "fused_pred": fused_pred
        }