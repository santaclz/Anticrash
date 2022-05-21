import sys
import torch
import numpy as np

CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
VEHICLES = [CAR, MOTORCYCLE, BUS, TRUCK]


def load_model():
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

    torch.save(model, 'model.pth')
    return model


def find_items(predictions, class_list, data):
    for elt in predictions:
        category = elt[5]
        if category in class_list:
            # print(f'found {results.names[int(category.item())]}')
            data.append(elt.tolist())


def main():
    data = []

    # Check if img path was given
    # if len(sys.argv) - 1 != 1:
    #     print('Usage: python3 object_detection.py <path_to_img>')
    #     exit(1)
    model = load_model()
    # img = sys.argv[1]
    img = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic3.bigstockphoto.com%2F0%2F9%2F3%2Flarge1500%2F390205342.jpg&f=1&nofb=1'

    # Inference
    # results = model(img)
    det_out, da_seg_out,ll_seg_out = model(img)

    print(1)
    # Parse results
    # predictions = results.pred[0]
    # find_items(predictions, VEHICLES, data)
    #
    # new_tensor = torch.Tensor(data)
    # results.pred[0] = new_tensor
    #
    # # Show results
    # results.show()
    #
    # Save results
    # results.save(save_dir='results/')


if __name__ == '__main__':
    main()
