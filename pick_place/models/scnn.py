import torch
import torchvision

class SiameseCNN(torch.nn.Module):
    def __init__(
        self,
        features="resnet",
        in_channels=2000,
        out_channels=3
    ):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        )
    
    def forward(self, x, ref=torch.zeros((1, 3, 1080, 1920))):
        x_emb = self.resnet(x)
        ref_emb = self.resnet(ref)

        full_emb = torch.cat([x_emb, ref_emb], dim=1)

        y_hat = self.fcn(full_emb)
        
        return y_hat