from PIL import Image
import sys
sys.path.append("/workspace/autonomous_driving/Part_1_Perception/9_ADAS/pytorch-nested-unet/")
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import archs
import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def yolo_model(model_path):
    yolo_model = YOLO(model_path)
    print('Yolo Model Load Success!!')
    return yolo_model

def unetplusplus_model(model_weight_path, yml_path, device):
    best_model = model_weight_path
    yml_path = yml_path

    with open(yml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    print("config_file:\n", data)
    model = model = archs.__dict__[data['arch']](data['num_classes'],data['input_channels'], data['deep_supervision'])
    model.to(device)
    model.load_state_dict(torch.load(best_model, map_location = device))
    print("Model Load Success!!")
    return model

def object_detection(model, ori_img):
    #ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    results = model(ori_img)

    detection_results = []
    for result in results:
        annotator = Annotator(ori_img)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]

            cls = int(box.cls)
            xmin = int(b[0])
            ymin = int(b[1])
            xmax = int(b[2])
            ymax = int(b[3])

            detection_results.append([xmin, ymin, xmax, ymax, cls])
    return detection_results


def line_segmentation(model, ori_img, device):
    ori_img = cv2.resize(ori_img, (512, 256))
    input = ori_img.astype('float32') / 255
    
    input = np.expand_dims(input, axis=0)
    input = torch.from_numpy(input).to(device)
    input = input.permute(0,3,1,2)
    output = model(input)
    output = torch.sigmoid(output)
    output = output.permute(0,2,3,1).cpu().detach()

    pred = np.array(output[0])*255
    pred_final = pred[:,:,0] + pred[:,:,1]
    pred_final = cv2.resize(pred_final, (ori_img.shape[1], ori_img.shape[0]))
    _, pred_final = cv2.threshold(pred_final, 240, 255, cv2.THRESH_BINARY)

    return pred_final

def inference_model(detection_model, segmentation_model, image_path, color_dict, device):
    ori_img = cv2.imread(image_path)
    detection_results = object_detection(detection_model, ori_img)
    pred_final = line_segmentation(segmentation_model, ori_img, device)

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    
    ## segmentation visualization
    pred_final = cv2.resize(pred_final, (ori_img.shape[1], ori_img.shape[0]), interpolation= cv2.INTER_NEAREST)
    roi_selection = ori_img[pred_final == 255]
    ori_img[pred_final == 255] = (0, 255 * 0.3, 0) + 0.6 * roi_selection

    ## detection visualization
    warning_status = False
    black_img = np.zeros_like(ori_img)
    for bbox in detection_results:
        cls = bbox[4]
        p_start, p_end = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        black_img = cv2.rectangle(black_img, p_start, p_end, color = color_dict[detection_model.names[cls]], thickness=-1)
        if bbox[3] > black_img.shape[0] * 0.95:
            warning_status = True
    
    final_result = cv2.addWeighted(black_img, 0.6, ori_img, 1.0, 0.0)
    if warning_status:
        cv2.rectangle(final_result, (0,0), (600, 140), (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(final_result, 'Warning!', (0, 100), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    
    return final_result

if __name__ == '__main__':
    #image_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/test/images/20201125_0_0_00_0_0_1_front_0027775.png"
    image_path  = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/test/images/20201125_0_0_00_0_0_1_front_0027701.png"
    save_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/infer_result/20201125_0_0_00_0_0_1_front_0027701.png"

    ## segmentation model Load
    model_weight_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/pytorch-nested-unet/models/lane_segmentation/model.pth"
    yml_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/pytorch-nested-unet/models/lane_segmentation/config.yml"
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    seg_model = unetplusplus_model(model_weight_path, yml_path, device)

    ## detection model load
    model_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/runs/detect/car_detection_s/weights/best.pt"
    detect_model = yolo_model(model_path)

    ## visualization
    color_dict = {'car':(255,0,0), 'pedestrian':(0, 0, 255)}
    final_result = inference_model(detect_model, seg_model, image_path, color_dict, device)
    
    final_result = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, final_result)