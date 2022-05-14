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
    argparser.add_argument(
        '-r',
        '--render',
        default=True,
        dest='render',
        type=bool,
        help='True/False, default = True')
    argparser.add_argument(
        '-l',
        '--list-towns',
        action='store_true',
        dest='list',
        help='List towns')
    argparser.add_argument(
        '-t',
        '--town',
        default='/Game/Carla/Maps/Town03',
        dest='town',
        type=str,
        help='Choose town (use -l to list the options)')
    args = argparser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()

    if not args.render:
        settings.no_rendering_mode = True
    else:
        settings.no_rendering_mode = False
    world.apply_settings(settings)

    if args.list:
        print(f'Available maps: {client.get_available_maps()}')
    if args.town:
        if not args.town in client.get_available_maps():
            print('This town does not exist. Run \'python3 init_script.py -l\' to see available towns.')
        else:
            world = client.load_world(args.town)
            world.unload_map_layer(carla.MapLayer.Buildings)
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)


if __name__ == '__main__':
    main()

