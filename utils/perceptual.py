
# --- Imports --- #
import torch
import torch.nn.functional as F


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)


class TotalLoss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(TotalLoss, self).__init__()
        self.loss_network = LossNetwork(vgg_model)

    def forward(self, pred_image, gt, lambda_loss):
        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = self.loss_network(pred_image, gt)
        total_loss = smooth_loss + lambda_loss * perceptual_loss

        return total_loss


