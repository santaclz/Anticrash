CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
VEHICLES = [CAR, MOTORCYCLE, BUS, TRUCK]

def find_items(results, class_list):
    data = []
    for i in range(results.pandas().xyxy[0].shape[0]):
        if results.pandas().xyxy[0].loc[i].at["class"] in class_list:
            xmin = results.pandas().xyxy[0].loc[i].at["xmin"]
            xmax = results.pandas().xyxy[0].loc[i].at["xmax"]
            ymin = results.pandas().xyxy[0].loc[i].at["ymin"]
            ymax = results.pandas().xyxy[0].loc[i].at["ymax"]
            confidence = results.pandas().xyxy[0].loc[i].at["confidence"]
            type = results.pandas().xyxy[0].loc[i].at["name"]
            data.append((xmin, xmax, ymin, ymax, confidence, type))
    return data

def get_vehicles(results, out_queue):
    #return find_items(results, VEHICLES)
    out_queue.put(find_items(results, VEHICLES))

