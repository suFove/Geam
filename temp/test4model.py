import torch

from trains.models import TextGraphFusionModule

x = torch.randn(3, 1, 2)
g = torch.randn(3, 1, 2)

# '''
#     对比方法
# '''
# x = x + g
# x = torch.concat(x, g, dim=1)
# x = torch.matmul(x, g.transpose(1, 2))
#
# '''
#     对比模型
# '''
#
# model = TextGraphFusionModule()
# model = FusionModule1()
# model = FusionModule2()
# model = FusionModule3()
#
# # ml, dl, transformer
# classifier = TextCNN()
# classifier = TextCNN1()
# classifier = TextCNN2()
# classifier = TextCNN3()
# classifier = LSTM1()
# classifier = LSTM2()
# classifier = LSTM3()
#
#
# classifier = BERT()

print(x)
print(g)
tgfm = TextGraphFusionModule()
# print(tgfm)
out1 = tgfm.forward(x, g)
print(out1.shape)
print(out1)

