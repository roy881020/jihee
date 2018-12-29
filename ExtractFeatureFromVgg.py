import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import torch.nn as nn

model = pretrainedmodels.__dict__['vgg16'](num_classes=1000, pretrained='imagenet')
print(model.eval())

load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)

path_img = 'input.jpg'
input_img = load_img(path_img)
input_tensor = tf_img(input_img)
input_tensor = input_tensor.unsqueeze(0)
input = torch.autograd.Variable(input_tensor, requires_grad=False)

output_logits = model(input)
print(output_logits.shape)

features = nn.Sequential(*list(model.children())[:-7])
out = features[0][:-11](input)
print(out.shape)