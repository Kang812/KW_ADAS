import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def segmentation_map(root_folder, save_folder):
    file_list = os.listdir(os.path.join(root_folder, 'train'))

    if not os.path.exists(os.path.join(save_folder, "lane/images")):
        os.makedirs(os.path.join(save_folder, "lane/images"))
    
    if not os.path.exists(os.path.join(save_folder, "lane/masks/0")):
        os.makedirs(os.path.join(save_folder, "lane/masks/0"))
    
    if not os.path.exists(os.path.join(save_folder, "lane/masks/1")):
        os.makedirs(os.path.join(save_folder, "lane/masks/1"))
    
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        name, ext = file.split(".")
        shutil.copyfile(os.path.join(root_folder, 'train', file), os.path.join(save_folder, "lane/images", file))

        mask = cv2.imread(os.path.join(root_folder, 'train_label', name + "_label" + "." + ext), 0)
        labels = list(np.unique(mask))
        labels.remove(0)

        for i in labels:
            new_mask = np.zeros(mask.shape)
            if i == 1:
                new_mask[mask == i] = 255
                cv2.imwrite(os.path.join(save_folder, "lane/masks/0", file), new_mask)
            if i == 2:
                new_mask[mask == i] = 255
                cv2.imwrite(os.path.join(save_folder, "lane/masks/1", file), new_mask)

if __name__ == "__main__":
    root_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/segmentation/"
    save_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/pytorch-nested-unet/inputs/"
    segmentation_map(root_folder, save_folder)