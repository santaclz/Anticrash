"""
W = gas
A = left
S = reverse
D = right
SPACE = hand brake
ESC = quit program
T = toggle bounding boxes (traffic lights + vehicles)
"""

### imports
import carla
import weakref
import random
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import torch
import time
from threading import Thread
from queue import Queue
from torchvision import transforms

from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_t
from pygame.locals import K_f

import prepare_server
import detect_trafficLights
import detect_vehicles
import detect_drivable

### globals
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# Load model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
torch.save(model, 'model.pth')

### client class
class BasicSynchronousClient(object):
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True
        self.boundingBoxes = False

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        """

        camera_transform = carla.Transform(carla.Location(x=0.8, z=1.3), carla.Rotation(pitch=10))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

    def control(self, car):
        """
        W = gas
        A = left
        S = reverse
        D = right
        SPACE = hand brake
        ESC = quit program
        T = toggle bounding boxes
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        if keys[K_t]:
            self.boundingBoxes = True
        if keys[K_f]:
            self.boundingBoxes = False

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Set new image from camera sensor.
        capture serves as sync flag.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render_detected(self, display, boxes):
        for box in boxes:
            if box[4] < 0.5:
                continue
            if box[-1] == "trafficlight":
                pygame.draw.rect(display, box[5], (box[0],box[2],box[1]-box[0],box[3]-box[2]), 2)
            else:
                pygame.draw.rect(display, (255, 51, 204), (box[0],box[2],box[1]-box[0],box[3]-box[2]), 2)

    # def recognizeFromImage(img, out_queue):
    #     recognized =  model(img)
    #     out_queue.put(("results", recognized))

    def render(self, display):
        """
        Convert camera sensor to rgb numpy array and render the image.
        Detect objects from image, work with results
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

            if self.boundingBoxes:
                my_queue = Queue()
                allElements = []
                recognized_objects = model(array)
                trafficThread = Thread(target=detect_trafficLights.get_trafficlights, args=(array, recognized_objects, my_queue))
                trafficThread.start()
                vehicleThead = Thread(target=detect_vehicles.get_vehicles, args=(recognized_objects, my_queue))
                vehicleThead.start()

                trafficThread.join()
                vehicleThead.join()

                while not my_queue.empty():
                    allElements += my_queue.get()

                # recognized_objects = model(array)
                # trafficlights = detect_trafficLights.get_trafficlights(array, recognized_objects)
                # vehicles = detect_vehicles.get_vehicles(recognized_objects)
                # allElements = trafficlights + vehicles

                if len(allElements) > 0:
                    self.render_detected(display, allElements)
                
    def game_loop(self):
        try:
            pygame.init()

            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            settings = self.world.get_settings()
            settings.no_rendering_mode = True # disable server rendering for better pc performance
            settings.fixed_delta_seconds = 0 # set fixed server step for synchronous client
            self.world.apply_settings(settings)

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()
            self.set_synchronous_mode(True)

            while True:
                self.world.tick() # prompt server for next frame

                self.capture = True
                pygame_clock.tick_busy_loop(60) # set MAX fps

                self.render(self.display)

                pygame.display.flip()
                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


### main
def main():
    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()