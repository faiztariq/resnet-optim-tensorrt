import torch.onnx
import torchvision
import torch

batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, device=torch.device("cuda"))
model_name = "img_classification_optimize.onnx"

# Export the model
torch.onnx.export(learn.model.cuda(),               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  model_name,   # destination of the model
                 )