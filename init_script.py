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

import carla
import random
import argparse

def main():
    argparser = argparse.ArgumentParser(description='Init settings')
    argparser.add_argument('-r', '--render', action='store_true', dest='render', help='If this flag is used then it will render')
    argparser.add_argument('-l', '--list-towns', action='store_true', dest='list', help='List towns')
    argparser.add_argument('-t', '--town', dest='town', type=str, help='Choose town (use -l to list the options)')
    args = argparser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()

    if not args.render:
        print('### Rendering disabled')
        settings.no_rendering_mode = True
    else:
        print('### Rendering enabled')
        settings.no_rendering_mode = False
    world.apply_settings(settings)

    if args.list:
        print(f'### Available maps:\n{client.get_available_maps()}')
    if args.town:
        if not args.town in client.get_available_maps():
            print('### This town does not exist. Run \'python3 init_script.py -l\' to see available towns.')
        else:
            print(f'### Loading new world...')
            world = client.load_world(args.town)
            world.unload_map_layer(carla.MapLayer.Buildings)
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)
            print('### Done')



if __name__ == '__main__':
    main()

