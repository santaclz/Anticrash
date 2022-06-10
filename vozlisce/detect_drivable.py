import torch
import numpy as np
from torchvision import transforms

#import intel_extension_for_pytorch as ipex

def init_model():
    model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
    torch.save(model, 'hybridnets.pth')
    #model = model.to(memory_format=torch.channels_last)
    model.eval()
    #model = ipex.optimize(model, dtype=torch.float32)

    return model

def detect_drivable_area(input_img, model):
    img = np.copy(input_img)
    img = np.float32(img) / 255.
    img = torch.tensor(img).permute((2,0,1)).unsqueeze(0) # tensor shape (1,3,640,384)

    img = img.to(memory_format=torch.channels_last)
    with torch.no_grad():
        features, regression, classification, anchors, segmentation = model(img)

    npimg = segmentation.detach().numpy()[0]

    return np.transpose(npimg, (1, 2, 0)) * 255
