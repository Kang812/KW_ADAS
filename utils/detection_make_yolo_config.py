import yaml
import os

def yolo_config(root_folder, nc, names, save_path):
    data = dict()
    data['train'] = os.path.join(root_folder, 'train')
    data['val'] = os.path.join(root_folder, 'val')
    data['test'] = os.path.join(root_folder, 'test')

    data['nc'] = nc
    data['names'] = names

    with open(save_path, "w") as f:
        yaml.dump(data, f)

if __name__ == '__main__':
    root_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/"
    nc = 2
    names =['car','pedestrian']
    save_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/yolo_config/car_detection.yaml"
    yolo_config(root_folder, nc, names, save_path)