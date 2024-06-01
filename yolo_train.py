from ultralytics import YOLO

model = YOLO('yolov8s.yaml')
results = model.train(data = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/yolo_config/car_detection.yaml",
                      epochs = 200,
                      batch = 32,
                      device = [0,1],
                      patience = 30,
                      name = 'car_detection_s')
