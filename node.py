# ==============================================================================
# -- Carla Setup ----------------------------------------------------------
# ==============================================================================
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- Program ----------------------------------------------------------
# ==============================================================================
import carla
import pygame
import random
import queue
import threading
import time
import examples.manual_control #import HUD, KeyboardControl
import m



WIDTH = 1280
HEIGHT = 720



def fetch_key_input():
    ''' Gets keys pressed '''
    pygame.init()
    actors = []
    world = None

    try:
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Get vehicle
        bp_vehicle = blueprint_library.filter('vehicle')[0]
        location = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp_vehicle, location)
        actors.append(vehicle)
        # Get camera
        bp_camera = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(bp_camera, camera_transform, attach_to=vehicle)
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        actors.append(camera)
        # Save them to our list of actors
        actors.extend([bp_vehicle, bp_camera])


        client.set_timeout(20.0)
        sim_world = client.get_world()
        display = pygame.display.set_mode(
                (WIDTH, HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = examples.manual_control.HUD(WIDTH, HEIGHT)
        # controller = examples.manual_control.KeyboardControl(world, False)

        clock = pygame.time.Clock()
        while True:
            img = image_queue.get()
            data = m.to_bgra_array(img)
            print(data)

            # call object_detection here

            clock.tick_busy_loop(60)

            # if controller.parse_events(client, world, clock, False):
            #     return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    except KeyboardInterrupt:
        print('\nProgram stopped by the user. Arrivederci')

    pygame.quit()



def main():
    # Run thread to get user input via PyGame
    # user_input = threading.Thread(target=fetch_key_input)
    # user_input.start()
    fetch_key_input()


    while True:
        pass
        # try:
        # TODO: ...
        #
        # except KeyboardInterrupt:
        #     print('\nCancelled by user. Bye!')


    # Cleanup
    for a in actors:
        a.destroy()


if __name__ == '__main__':
    main()
