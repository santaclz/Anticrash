import torch
import numpy as np
from torchvision import transforms


def init_model():
    model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
    torch.save(model, 'hybridnets.pth')
    #model = torch.load('hybridnets.pth')
    #model.eval()
    return model

def detect_drivable_area(input_img):
    model = init_model()
    #img = Image.open("./results/carla-semafor.jpg").resize((640,384), Image.ANTIALIAS)
    #img = np.copy(input_img[:,:,1:4])
    img = np.copy(input_img)
    #img = img[:,:,1:4]
    #img.resize((640,384,3))
    #print(img.shape)
    img = torch.tensor(img).permute((2,0,1)).unsqueeze(0) # tensor shape (1,3,640,384)
    #img = torch.from_numpy(img).permute((2,0,1)).unsqueeze(0) # tensor shape (1,3,640,384)
    img = img.float() / 255

    features, regression, classification, anchors, segmentation = model(img)

    npimg = segmentation.detach().cpu().numpy()[0]
    return np.transpose(npimg, (1, 2, 0)) * 255
