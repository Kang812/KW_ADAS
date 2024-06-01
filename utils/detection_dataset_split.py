import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def dataset_split(root_folder, save_folder, valid_size = 0.2, test_size = 0.5, seed = 42):
    file_list = os.listdir(os.path.join(root_folder, "images"))

    if not os.path.exists(os.path.join(save_folder, "train/images")):
        os.makedirs(os.path.join(save_folder, "train/images"))
    
    if not os.path.exists(os.path.join(save_folder, "train/labels")):
        os.makedirs(os.path.join(save_folder, "train/labels"))
    
    if not os.path.exists(os.path.join(save_folder, "val/images")):
        os.makedirs(os.path.join(save_folder, "val/images"))
    
    if not os.path.exists(os.path.join(save_folder, "val/labels")):
        os.makedirs(os.path.join(save_folder, "val/labels"))
    
    if not os.path.exists(os.path.join(save_folder, "test/images")):
        os.makedirs(os.path.join(save_folder, "test/images"))
    
    if not os.path.exists(os.path.join(save_folder, "test/labels")):
        os.makedirs(os.path.join(save_folder, "test/labels"))
    
    train, valid = train_test_split(file_list, test_size = valid_size, random_state = seed)
    valid, test = train_test_split(valid, test_size=test_size, random_state=seed)

    data_list = [train, valid, test]
    for i in range(len(data_list)):
        dataset = data_list[i]
        if i == 0:
            print("Train:")
            data_position = "train"
        elif i == 1:
            print("Val:")
            data_position = "val"
        elif i == 2:
            print("Test:")
            data_position = "test"

        for i in tqdm(range(len(dataset))):
            shutil.copy(os.path.join(root_folder, "images", dataset[i]), os.path.join(save_folder, data_position, "images", dataset[i]))
            shutil.copy(os.path.join(root_folder, "new_labels", dataset[i].split(".")[0] + ".txt"), os.path.join(save_folder, data_position, "labels", dataset[i].split(".")[0] + ".txt"))

if __name__ == "__main__":
    root_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/"
    save_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/" 
    valid_size = 0.2 
    test_size = 0.5
    dataset_split(root_folder, save_folder, valid_size = 0.2, test_size = 0.5, seed = 42)

