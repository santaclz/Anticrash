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
#import torch
import matplotlib.pyplot as plt
import object_detection

import lane_detection


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

    #vehicle.set_autopilot(True)

    # Get camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    actor_list.append(camera)


    # run one step of simulation
    #world.tick()

    # Init YOLOP
    #model = torch.hub.load('hustvl/yolop', 'yolop', trust_repo=True, pretrained=True)

    model = object_detection.load_model()

    while True:
        image = image_queue.get()
        #image.save_to_disk("test-%06d.png" % (image.frame))
        data = to_bgra_array(image)
        #det_out, da_seg_out, ll_seg_out = model(data)
        try:
            # data_lane = lane_detection.draw_lane(data)
            objects = object_detection.get_object(model, data)
            cv2.imshow("frame", data)
        except Exception:
            pass
        #time.sleep(1)

        #if cv2.waitKey(1) & 0xFF == ord('w'):
            #vehicle.set_target_velocity(carla.Vector3D(-60,0,0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for a in actor_list:
        a.destroy()

