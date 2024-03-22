import torch
import torch.nn
    
class SCNNLoss(torch.nn.Module):
    def __init__(self, w=0.99):
        super().__init__()

        self.w = w

    def forward(self, y, y_hat):
        q = y[2]
        q_hat = y_hat[2]
        criterion = torch.nn.MSELoss()
        RMSY = torch.sqrt(criterion(y[:2], y_hat[:2]))
        RMSR = torch.sqrt(criterion(q, q_hat))
        return self.w * RMSY + (1 - self.w) * RMSR