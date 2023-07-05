import torch
import torchvision

def get_model(net_name, hidden_dim=50):
    model = None
    if net_name == "resnet18in":
        model = torchvision.models.resnet.ResNet(
                            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=hidden_dim, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=torch.nn.InstanceNorm2d
                            )
    elif net_name == "resnet18bn":
        model = torchvision.models.resnet.ResNet(
                            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=hidden_dim, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=torch.nn.BatchNorm2d
                            )
    elif net_name == "resnet18ft":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_feat = model.fc.in_features
        model.fc = torch.nn.Linear(num_feat, hidden_dim)
    elif net_name == "mobilenetv3s":
        model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(576, hidden_dim)
    elif net_name == "mobilenetv3sft":
        model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Linear(576, hidden_dim)
    else:
        raise Exception("Incorrect MNet name.")
    return model

class Mnet(torch.nn.Module):
    def __init__(self, net_name="mobilenetv3s", hidden_dim=50):
        super().__init__()
        stain_dim = 3
        self.model = get_model(net_name, hidden_dim)

        self.M_mean = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(hidden_dim, 2*stain_dim),
        )

        self.M_log_var = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, H, W)
        Returns:
            mean: (batch_size, stain_dim, 2)
            var: (batch_size, 1, 2)
        """    
        x = torch.nn.functional.instance_norm(x)
        x = self.model(x)
        mean = self.M_mean(x) # (batch_size, 2*stain_dim)
        mean = mean.view(mean.shape[0], 3, 2) # (batch_size, stain_dim, 2)
        l1 = torch.norm(mean, dim=1, keepdim=True) # (batch_size, )
        mean = torch.div(mean , l1 + 1e-10) # (batch_size, stain_dim, 2)

        var = torch.exp(self.M_log_var(x)) + 1e-10 # (batch_size, 2)
        var = var.view(var.shape[0], 1, 2) # (batch_size, 1, 2)

        return mean, var