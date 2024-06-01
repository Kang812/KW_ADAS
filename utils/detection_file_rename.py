import os
import shutil
from tqdm import tqdm

# 한국 이름을 지우기 위한 코드
def rename(root_folder, change_folder):
    city = ['서울특별시']
    gu = ['영등포구', '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구', '노원구','은평구', '서대문구', '마포구', '양천구', '강서구', '구로구','금천구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구']
    weather = ['맑음']
    gan = ['주간', '야간']
    sil = ['실내', '실외']

    file_list =  os.listdir(root_folder)

    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        name = file.split(".")[0]
        ext = file.split(".")[1]
        p = name.split("_")

        p[1] = str(city.index(p[1]))
        p[3] = f"{gu.index(p[3]):02d}"
        p[4] = str(weather.index(p[4]))
        p[5] = str(gan.index(p[5]))
        p[6] = str(sil.index(p[6]))
        new_name = "_".join(p) + '.' + ext
        
        if ext == "json":
            if not os.path.exists(os.path.join(change_folder, "labels")):
                os.makedirs(os.path.join(change_folder, "labels"))
            shutil.copyfile(os.path.join(root_folder, file), os.path.join(change_folder, "labels", new_name))
        else:
            if not os.path.exists(os.path.join(change_folder, "images")):
                os.makedirs(os.path.join(change_folder, "images"))
            shutil.copyfile(os.path.join(root_folder, file), os.path.join(change_folder, "images", new_name))

if __name__ == '__main__':
    root_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/bounding_box/"
    change_folder = "/workspace/autonomous_driving/Part_1_Perception/9_ADAS/data/detection/"
    rename(root_folder, change_folder)