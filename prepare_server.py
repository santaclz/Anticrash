import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

settings = world.get_settings()
settings.no_rendering_mode = True # disable server rendering for better pc performance
settings.fixed_delta_seconds = 0 # set fixed server step for synchronous client
world.apply_settings(settings)