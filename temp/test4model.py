import torch

from trains.models import TextGraphFusionModule


x = torch.randn(3, 1, 2)
y = torch.randn(3, 1, 2)

print(x)
tgfm = TextGraphFusionModule()
# print(tgfm)
out1 = tgfm.forward(x, y)
print(out1.shape)
print(out1)

