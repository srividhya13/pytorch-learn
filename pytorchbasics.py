import torch
import torch.nn

class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M, N = to_reshape.shape
        rs = torch.reshape(to_reshape, (M * N // 2, 2))
        return torch.round(rs, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        a = torch.mean(to_avg, dim = 0)
        return torch.round(a, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        c = torch.cat((cat_one, cat_two), dim = 1)
        return torch.round(c, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        l = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(l, decimals=4)
