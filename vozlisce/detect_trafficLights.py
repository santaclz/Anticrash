import numpy as np
import skimage.exposure as exposure
import cv2

def detectDominantLight(image):
    original = image.copy()

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

    # Find color
    threshCount = 100
    if redCount > greenCount:
        color = (204, 0, 0)
    elif greenCount > redCount:
        color = (0, 255, 0)
    else:
        color = (0, 0, 204)
    #elif redCount < threshCount and greenCount < threshCount:
    #    color = (255, 255, 102)
    #else:
    #    color = (0, 0, 204)
    return color

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

def getTrafficLightStates(trafficLights, image):
    fullImage = np.copy(image)
    trafficLightStates = list()
    for light in trafficLights:
        horizontal = (light[1]-light[0])//4
        vertical = (light[3]-light[2])//10
        croppedImage = fullImage[int(light[2]+vertical):int(light[3]-vertical), int(light[0]+horizontal):int(light[1]-horizontal)]
        trafficLightStates.append((light[0], light[1], light[2], light[3], light[4], detectDominantLight(croppedImage), "trafficlight"))
    return trafficLightStates

def get_trafficlights(img, recognized, out_queue):
    extracted_trafficlights = extractTrafficLights(recognized)
    if len(extracted_trafficlights) == 0:
        return []
    #return getTrafficLightStates(extracted_trafficlights, img[:, :, ::-1])
    out_queue.put(getTrafficLightStates(extracted_trafficlights, img[:, :, ::-1]))