import numpy as np
import argparse
import pygame
import skimage.exposure as exposure
import cv2
import torch

'''
usage: detect_trafficLights.py -i <path_to_img> (needed) -v True (optional, show inbetween steps)
'''

def detectDominantLight(image, radius, print_verbose):
    # open image, make a copy and grayscale
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate 2D histograms for pairs of channels: GR
    histGR = cv2.calcHist([original], [1, 2], None, [256, 256], [0, 256, 0, 256])
    
    # histogram is float and counts need to be scale to range 0 to 255
    histScaled = exposure.rescale_intensity(histGR, in_range=(0,1), out_range=(0,255)).clip(0,255).astype(np.uint8)
    
    # make masks
    ww = 256
    hh = 256
    ww13 = ww // 3
    ww23 = 2 * ww13
    hh13 = hh // 3
    hh23 = 2 * hh13
    black = np.zeros_like(histScaled, dtype=np.uint8)

    # specify points in OpenCV x,y format
    ptsUR = np.array( [[[ww13,0],[ww-1,hh23],[ww-1,0]]], dtype=np.int32 )
    redMask = black.copy()
    cv2.fillPoly(redMask, ptsUR, (255,255,255))
    ptsBL = np.array( [[[0,hh13],[ww23,hh-1],[0,hh-1]]], dtype=np.int32 )
    greenMask = black.copy()
    cv2.fillPoly(greenMask, ptsBL, (255,255,255))

    #Test histogram against masks
    region = cv2.bitwise_and(histScaled,histScaled,mask=redMask)
    redCount = np.count_nonzero(region)
    region = cv2.bitwise_and(histScaled,histScaled,mask=greenMask)
    greenCount = np.count_nonzero(region)
    #print('redCount:',redCount)
    #print('greenCount:',greenCount)

    # Find color
    threshCount = 100
    if redCount > greenCount and redCount > threshCount:
        color = "RED"
    elif greenCount > redCount and greenCount > threshCount:
        color = "GREEN"
    elif redCount < threshCount and greenCount < threshCount:
        color = "YELLOW"
    else:
        color = "UNKNOWN"
    #print("color: ",color)
    return color

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, int(gray.shape[1]/2))
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     print(circles.shape)
    #     for (x, y, r) in circles:
    # 		# draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(original, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #         cv2.imshow("circle", original)
    #         cv2.waitKey(0)

    # get brightest area in image with a given radious
    # gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # image = original.copy()
    # detectedColor = image[maxLoc[1], maxLoc[0]] # center of detected peak brightness
    # if print_verbose == "True":
    #     print(f"detected(BGR): {detectedColor}")
    #     cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)
    #     cv2.imshow("Robust", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # if detectedColor[1] == max(detectedColor) and detectedColor[2] == min(detectedColor):
    #     return "GREEN"
    # elif detectedColor[2] == max(detectedColor) or detectedColor[2] > detectedColor[1]/1.15 and detectedColor[0] == min(detectedColor):
    #     return "RED"
    # else:
    #     return "UNKNOWN"

def extractTrafficLights(results):
    coords = list()
    for i in range(results.pandas().xyxy[0].shape[0]):
        # Get data about all the recognized traffic lights
        if results.pandas().xyxy[0].loc[i].at["class"] == 9: # index 9 is traffic light
            xmin = results.pandas().xyxy[0].loc[i].at["xmin"]
            xmax = results.pandas().xyxy[0].loc[i].at["xmax"]
            ymin = results.pandas().xyxy[0].loc[i].at["ymin"]
            ymax = results.pandas().xyxy[0].loc[i].at["ymax"]
            confidence = results.pandas().xyxy[0].loc[i].at["confidence"]
            coords.append((xmin, xmax, ymin, ymax, confidence))
    return coords

def getTrafficLightStates(trafficLights, image, print_verbose):
    fullImage = cv2.imread(image)
    trafficLightStates = list()
    for light in trafficLights:
        width = int(light[1] - light[0])
        height = int(light[3] - light[2])
        #if fullImage.shape[1] > fullImage.shape[0]: # vertically orented
        #    croppedImage = fullImage[int(light[2])+int(height/8):int(light[3])-int(height/8), int(light[0])+int(width/3):int(light[1])-int(width/3)]
        #else: # horizontally oriented
        #    croppedImage = fullImage[int(light[2]):int(light[3]), int(light[0]):int(light[1])]
        croppedImage = fullImage[int(light[2]):int(light[3]), int(light[0]):int(light[1])]
        if print_verbose == "True":
            cv2.imshow('cropped', croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        shorter = min(width, height)
        radius = int(shorter/7) if int(shorter/7) % 2 != 0 else int(shorter/7) + 1
        trafficLightStates.append((light[0], light[1], light[2], light[3], light[4], detectDominantLight(croppedImage, radius, print_verbose)))
    return trafficLightStates

def drawFoundLightsAndStates(trafficLights, image):
    pygame.init()
    pyImage = pygame.image.load(image)
    screen = pygame.display.set_mode(pyImage.get_rect().size)
    font = pygame.font.Font('freesansbold.ttf', 16)
    screen.blit(pyImage, (0,0))

    for light in trafficLights:
        height = light[1]-light[0]
        width = light[3]-light[2]
        if light[5] == "RED":
            borderColor = (255, 0, 0)
        elif light[5] == "GREEN":
            borderColor = (0, 255, 0)
        elif light[5] == "YELLOW":
            borderColor = (255, 255, 102)
        else:
            borderColor = (0, 0, 255)
        pygame.draw.rect(screen, borderColor, pygame.Rect(light[0], light[2], height, width), 3)
        pygame.draw.rect(screen, borderColor, pygame.Rect(light[0], light[2]-16, 16*2, 16))
        text = font.render(str(round(100*light[4])) +"%" , True , (0,0,0))
        screen.blit(text , (light[0],light[2]-16))

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == 27):
                #shutil.rmtree("temporary/", ignore_errors=False, onerror=None)
                quit()

def main():
    # read given arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    ap.add_argument("-v", "--verbose", help = "show images as they are being processed")
    args = vars(ap.parse_args())

    # check if image is provided
    if args["image"] is None:
        print("No image provided")
        quit()

    # load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n, yolov5x6, custom
    torch.save(model, 'model.pth')

    # recognize all traffic lights in image
    results = model(args["image"])

    #save cropped image of every detected traffic light
    #results.crop(True, "temporary/")

    # extract only traffic lights from the result
    trafficLights = extractTrafficLights(results)

    # check for state of every detected traffic light
    trafficLightStates = getTrafficLightStates(trafficLights, args["image"], args["verbose"])

    # draw results
    drawFoundLightsAndStates(trafficLightStates, args["image"])

if __name__ == '__main__':
    main()