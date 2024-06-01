from ultralytics import YOLO

def detection_eval(best_model_path, batch, split, device, name):
    model = YOLO(best_model_path)
    metrics = model.val(batch = batch, split = split, device = device, name = name)

if __name__ == '__main__':
    best_model_path = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/runs/detect/car_detection_s/weights/best.pt"
    batch = 32
    split = "test"
    device = [0,1]
    name = 'car_detection_testset_eval'
    
    detection_eval(best_model_path, batch, split, device, name)