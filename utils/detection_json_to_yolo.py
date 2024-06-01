import json
import os
from tqdm import tqdm
import numpy as np

def json_to_yolo_convert(root_folder, save_folder):
    classes = ['car', 'pedestrian']
    cars = ['일반차량', '목적차량(특장차)', '이륜차']

    file_list = os.listdir(root_folder)
    
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        name, ext = file.split(".")
        
        with open(os.path.join(root_folder, file), "r") as f:
            data = json.load(f)
            annotations = data['annotations']
        
        width = int(data['camera']['resolution_width'])
        height = int(data['camera']['resolution_height'])

        f = open(os.path.join(save_folder, name + ".txt"), 'a')
        
        for annotation in annotations:
            anno_label = annotation['label']
            points = np.array(annotation['points']).ravel()
            points_x = [points[k] for k in range(len(points)) if k % 2 == 0]
            points_y = [points[k] for k in range(len(points)) if k % 2 == 1]

            xmin = min(points_x)
            ymin = min(points_y)
            xmax = max(points_x)
            ymax = max(points_y)

            w = xmax - xmin
            h = ymax - ymin

            center_x = (xmin + w/2)/width
            center_y = (ymin + h/2)/height
            w = w/width
            h = h/height

            if anno_label in cars:
                label_idx = 0
            elif anno_label == '보행자':
                label_idx = 1
            
            yolo_format = "%s %s %s %s %s\n" % (label_idx, center_x, center_y, w, h)
            f.write(yolo_format)
        f.close()

if __name__ == '__main__':
    root_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/labels"
    save_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/new_labels"
    json_to_yolo_convert(root_folder, save_folder)