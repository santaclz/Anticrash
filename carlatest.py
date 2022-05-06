import carla
import random

#client = carla.Client('192.168.8.134', 2000)
client = carla.Client('localhost', 2000)
world = client.get_world()


# Get the blueprint library and filter for the vehicle blueprints
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn 50 vehicles randomly distributed throughout the map 
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))


ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
