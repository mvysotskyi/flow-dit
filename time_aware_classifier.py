import torch
import torch.nn as nn
from torchvision.models import resnet18


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class TimeAwareResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, time_emb_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
        )

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        t_emb = self.time_mlp(t)
        h = feat + t_emb
        logits = self.classifier(h)
        return logits



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    checkpoint_path = ""

    model = TimeAwareResNetClassifier(num_classes=num_classes, time_emb_dim=128, pretrained=False)
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    x = torch.randn(4, 3, 224, 224, device=device)
    t = torch.randint(0, 1000, (4,), device=device)
    with torch.no_grad():
        logits = model(x, t)
    print(logits.shape)
