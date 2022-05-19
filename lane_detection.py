import carla
import random
import queue
import numpy as np
import cv2
import time
#import torch
import matplotlib.pyplot as plt

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

### Lane detection
def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    return np.array([left_line, right_line])

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            except Exception:
                print("Drawing line failed")
    return line_image

def draw_lane(image):
    # Detect sharp edges
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # Get region of interest
    height = image.shape[0]
    polygons = np.array([[(100, height), (700, height), (400, 310)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(canny, mask[:,:,0])
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    avg_lines = avg_slope_intercept(lane_image, lines)
    drawn_lines = draw_lines(image, avg_lines)
    result = cv2.bitwise_or(drawn_lines, image)
    #result = cv2.addWeighted(draw_lines(image, lines), 1, lane_image, 1, 1) 
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #plt.imshow(canny)
    #plt.show()
    return result


if __name__ == "__main__":
    client = carla.Client('192.168.8.134', 2000)
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

    while True:
        image = image_queue.get()
        #image.save_to_disk("test-%06d.png" % (image.frame))
        data = to_bgra_array(image)
        #det_out, da_seg_out, ll_seg_out = model(data)
        try:
            data_lane = draw_lane(data)
            cv2.imshow("frame", data_lane)
        except Exception:
            pass
        #time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            vehicle.set_target_velocity(carla.Vector3D(-60,0,0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for a in actor_list:
        a.destroy()

