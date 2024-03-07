import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision

def load_model(model_name):
    
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model


if __name__ == "__main__":
    scripted_model = load_model("resnet18")
    print(scripted_model)

