import os
import glob
import sys

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import random
import queue
import numpy as np
import cv2
import time
import torch
import matplotlib.pyplot as plt
import object_detection

import lane_detection
import detect_trafficLights
import drivable_detect

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n, yolov5x6, custom
torch.save(model, 'model.pth')

#model2 = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
#torch.save(model2, 'hybridnets.pth')

### Helper functions
def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array

def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def convert_rgb_bgr(array):
    """numpy array: RGB <=> BGR"""
    return array[:, :, ::-1].copy()

def recognizeFromImage(img):
    return model(img)

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    # Random vehicle
    #bp = random.choice(blueprint_library.filter("vehicle"))
    #print(blueprint_library.filter("vehicle"))
    bp = blueprint_library.filter("vehicle")[0]

    # Random spawn point
    transform = random.choice(world.get_map().get_spawn_points()) 

    # Spawn a vehicle
    actor_list = []
    vehicle = world.spawn_actor(bp, transform) 
    actor_list.append(vehicle)

    vehicle.set_autopilot(True)

    # Get camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '384')
    
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.LifoQueue()
    camera.listen(image_queue.put)
    actor_list.append(camera)

    model = object_detection.load_model()

    # Main loop
    while True:
        image = image_queue.get()
        data = to_rgb_array(image)

        bgr_img = to_rgb_array(image)
        rgb_img = convert_rgb_bgr(bgr_img)
        recognized_objects = recognizeFromImage(rgb_img)
        
        ### TODO run in another thread?
        data_drivable = drivable_detect.detect_drivable_area(data)
        data_drivable = data_drivable.astype('uint8')
        tmp_data = data.astype('uint8')

        data_drivable = convert_rgb_bgr(data_drivable)
        data_drivable[:,:,2] = 0
        tmp_data = convert_rgb_bgr(tmp_data)
        #data_together = cv2.bitwise_or(data_drivable, tmp_data)
        try:
            # data_lane = lane_detection.draw_lane(data)
            objects = object_detection.get_object(model, data)
            #cv2.imshow("frame", data_together)
            #data_lane = lane_detection.draw_lane(data)
            #cv2.imshow("frame", data_lane)

            # drawn_trafficlights = detect_trafficLights.get_trafficlights_drawn(rgb_img, bgr_img, recognized_objects)
            # data_together = cv2.bitwise_or(drawn_trafficlights, data_drivable)
            #cv2.imshow("frame", drawn_trafficlights)

        except Exception:
            # Draw raw image if everything fails
            data_together = cv2.bitwise_or(tmp_data, data_drivable)

            print("exception caught!")

        # cv2.imshow('frame', objects)
        cv2.imshow("frame", data_together)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for a in actor_list:
        a.destroy()

