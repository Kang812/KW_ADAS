# KW_ADAS
Advanced Dirver Assistance System

## 필요 library 설치
```
pip install -r requirements.txt
```

## 사용한 데이터 셋
-[line segmentation](https://www.kaggle.com/datasets/thomasfermi/lane-detection-for-carla-driving-simulator)

-[detection](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=195)

## 데이터 전처리
- detection
```
python ./utils/detection_file_rename.py
python ./utils/detection_json_to_yolo.py
python ./utils/detection_dataset_split.py
```

- segmentation
```
python ./utils/segmentation_map.py
```
## yolo config file
```
python ./utils/detection_make_yolo_config.py
```

## model train
- detection
```
python yolo_train.py
```

- segmentation
```
python ./pytorch-nested-unet/train.sh
```
## model eval
- detection
```
python yolo_eval.py
```
| Model                           |  mAP50  |Download |
|:------------------------------- |:-------:|:-------:|
| yolov8s                         |  0.852  |  0.086  |

- segmentation
```
python ./pytorch-nested-unet/val.py --name lane_segmentation
```

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net++                         |  0.852  |  0.086  |

## Result
```
python inference.py
```

![Demo](./infer_result/20201125_0_0_00_0_0_1_front_0027701.png)
