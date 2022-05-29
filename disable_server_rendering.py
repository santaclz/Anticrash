import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

settings = world.get_settings()
settings.no_rendering_mode = True
world.apply_settings(settings)