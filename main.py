import carla
import random
import queue
import numpy as np
import cv2
import time
import torch
import matplotlib.pyplot as plt

import lane_detection
import detect_trafficLights

# load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n, yolov5x6, custom
torch.save(model, 'model.pth')

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

    #vehicle.set_autopilot(True)

    # Get camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    actor_list.append(camera)


    # run one step of simulation
    #world.tick()

    # Init YOLOP
    #model = torch.hub.load('hustvl/yolop', 'yolop', trust_repo=True, pretrained=True)

    while True:
        image = image_queue.get()
        #image.save_to_disk("test-%06d.png" % (image.frame))
        data = to_bgra_array(image)
        #det_out, da_seg_out, ll_seg_out = model(data)

        bgr_img = to_rgb_array(image)
        rgb_img = convert_rgb_bgr(bgr_img)
        recognized_objects = recognizeFromImage(rgb_img)
        
        try:
            #data_lane = lane_detection.draw_lane(data)
            #cv2.imshow("frame", data_lane)
            drawn_trafficlights = detect_trafficLights.get_trafficlights_drawn(rgb_img, bgr_img, recognized_objects)
            cv2.imshow("frame", drawn_trafficlights)
        except Exception:
            print("exception caught!")
        
        #time.sleep(1)
        #if cv2.waitKey(1) & 0xFF == ord('w'):
            #vehicle.set_target_velocity(carla.Vector3D(-60,0,0))
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    for a in actor_list:
        a.destroy()

