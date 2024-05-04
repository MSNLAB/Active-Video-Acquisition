import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, temperature=1.0, backbone="transformer"):
        super(ActorCritic, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.temperature = temperature

        if backbone == "lstm":
            self.backbone_net = nn.LSTM(in_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
            self.hidden_dim *= 2

        elif backbone == "transformer":
            encoder = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=1, dim_feedforward=hidden_dim, dropout=0.0, batch_first=True
            )
            self.backbone_net = nn.Sequential(
                nn.TransformerEncoder(encoder, num_layers=2),
                nn.Linear(in_dim, hidden_dim),
            )

        else:
            raise NotImplementedError(f"given backbone {backbone} is not supported.")

        self.pi_net = nn.Linear(self.hidden_dim, 2)
        self.v_net = nn.Linear(self.hidden_dim, 1)

    def _backbone_forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        if self.backbone == "lstm":
            x, _ = self.backbone_net(x)

        elif self.backbone == "transformer":
            x = self.backbone_net(x)

        return x

    def pi(self, x):
        x = self._backbone_forward(x)
        x = self.pi_net(x)

        return torch.softmax(x / self.temperature, dim=-1)

    def v(self, x):
        x = self._backbone_forward(x)
        x = self.v_net(x).squeeze(-1)

        return torch.mean(x, dim=-1)
