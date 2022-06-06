import torch
import numpy as np

CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
VEHICLES = [CAR, MOTORCYCLE, BUS, TRUCK]


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    torch.save(model, 'model.pth')
    return model


def find_items(predictions, class_list, data):
    for elt in predictions:
        category = elt[5]
        if category in class_list:
            print(f'found car: {elt[1:5]}')
            data.append(elt.tolist())



def get_object(model, img):
    data = []

    # Inference
    results = model(img)

    # Parse results
    predictions = results.pred[0]
    find_items(predictions, VEHICLES, data)

    return np.array(data)

    # new_tensor = torch.Tensor(data)
    # results.pred[0] = new_tensor

    # Show results
    # results.show()
    # return results

    # Save results
    # results.save(save_dir='results/')

