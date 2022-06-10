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
import copy
import cv2
import time
import torch
import matplotlib.pyplot as plt
import object_detection

import lane_detection
import detect_trafficLights
import drivable_detect

WIDTH = 640
HEIGHT = 384
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

    ### CAMERA
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', f'{WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{HEIGHT}')
    camera_bp.set_attribute('fov', 90)

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.LifoQueue()
    camera.listen(image_queue.put)
    actor_list.append(camera)

    ### LIDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    lidar_bp.set_attribute('rotation_frequency', '10')

    lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_queue = queue.LifoQueue()
    lidar.listen(lidar_queue.put)
    actor_list.append(lidar)
    

    # Main loop
    while True:
        vehicles = []

        ### LIDAR logic
        lidar_measurement = lidar_queue.get()
        for location in lidar_measurement:
            print(location)

        '''
        lidar_range = 20 
        points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('float32'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(WIDTH, HEIGHT) / (2.0 * lidar_range)
        lidar_data += (0.5 * HEIGHT, 0.5 * WIDTH)
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (HEIGHT, WIDTH, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        
        lidar_img = np.flip(lidar_img, axis=0)

        cv2.imshow("lidar_frame", lidar_img)
        '''

        ### IMG logic
        image = image_queue.get()
        data = to_rgb_array(image)

        bgr_img = to_rgb_array(image)
        rgb_img = convert_rgb_bgr(bgr_img)
        recognized_objects = recognizeFromImage(rgb_img)

        object_detection.find_items(recognized_objects.pred[0], object_detection.VEHICLES, vehicles)
        
        ### TODO run in another thread?
        data_drivable = drivable_detect.detect_drivable_area(data)
        data_drivable = data_drivable.astype('uint8')
        tmp_data = data.astype('uint8')

        data_drivable = convert_rgb_bgr(data_drivable)
        data_drivable[:,:,2] = 0
        tmp_data = convert_rgb_bgr(tmp_data)
        #data_together = cv2.bitwise_or(data_drivable, tmp_data)
        objects = object_detection.get_object(model, data)
        try:
            ### Detect lines
            data_lane = copy.deepcopy(data_drivable)
            data_lane[:,:,1] = 0
            #data_lane_lines = lane_detection.draw_lane(data_lane)
            #cv2.imshow("frame2", data_lane_lines)
            cv2.imshow("frame2", data_lane)
            #cv2.imshow("frame", data_together)


            drawn_trafficlights = detect_trafficLights.get_trafficlights_drawn(rgb_img, bgr_img, recognized_objects)
            data_together = cv2.bitwise_or(drawn_trafficlights, data_drivable)
            #cv2.imshow("frame", drawn_trafficlights)

        except Exception:
            # Draw raw image if everything fails
            data_together = cv2.bitwise_or(tmp_data, data_drivable)

            print("exception caught!")

        cv2.imshow("frame", data_together)
        #cv2.imshow("frame2", data_lane)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for a in actor_list:
        a.destroy()

