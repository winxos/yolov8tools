'''
yolov8 data spliter
winxos 20231213
### usage
split directory :

projs/
-yolov8_spliter.py
-proj1/
--images/
--labels/

to:

-proj1/
--images/
--labels/
--train/
---images/
---labels/
--valid/
---images/
---labels/

with the setting ratio.
### this script will keep your raw data.
'''
import argparse
import os
from pathlib import Path,PurePath
import random
import shutil 
from tqdm import tqdm
IMG_FORMATS = 'bmp', 'jpeg', 'jpg'
def move_data(path,dst,tags):
    img_dir = f"{path}/{dst}/images"
    label_dir = f"{path}/{dst}/labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)  
    for t in tqdm(tags, desc=f"moving {dst}", ascii=" =", colour="green"):
        try:
            img = f"{path}/images/{t}.jpg"
            label = f"{path}/labels/{t}.txt"
            img_dst = f"{img_dir}/{t}.jpg"
            label_dst = f"{label_dir}/{t}.txt"
            shutil.copy(img,img_dst)
            shutil.copy(label,label_dst)
        except:
            print(f"copy err. {t}")
    print("done.")
def move_data_cls(path,dst,tags):
    dst_dir = f"{path}/{dst}"
    os.makedirs(dst_dir, exist_ok=True)
    for img in tqdm(tags, desc=f"moving {dst}", ascii=" =", colour="green"):
        try:
            a_path = PurePath(img)
            subd = str(a_path.relative_to(f"{path}/images")).split("\\")[0]
            fn = "-".join(str(a_path.relative_to(f"{path}/images")).split("\\")[1:])
            os.makedirs(f"{dst_dir}/{subd}", exist_ok=True)
            dst = f"{dst_dir}/{subd}/{fn}"
            shutil.copy(img,dst)
        except:
            print(f"copy err. {img}")
    print("done.")  
def main():
    parser = argparse.ArgumentParser(description="yolov8 converter")
    parser.add_argument("--proj",default="hand",help="project dir name")
    parser.add_argument("--proportion",default=0.8,type=float,help="train data proportion,default 0.8, [0.0 - 1.0]")
    parser.add_argument("--shuffle",default=0,type=int,help="random split the dataset,default 0, [0|1]")
    parser.add_argument("--mode",default="seg",help="mode:seg,cls. cls no need labels")
    args = parser.parse_args()
    print(args)
    PROJ = args.proj
    if args.shuffle == 0:
        random.seed(42)
    imgs = f"{PROJ}/images"
    if args.mode == "cls":
        path = Path(imgs)  # images dir
        files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
        print(f"total images: {len(files)}")
        random.shuffle(files)
        print(f"shuffled, the first: {files[0]}")
        train_size = int(args.proportion * len(files))
        valid_size = len(files) - train_size
        move_data_cls(PROJ,"train",files[:train_size])
        move_data_cls(PROJ,"val",files[train_size:])
    
    elif args.mode == "seg":
        labels = f"{PROJ}/labels"
        tag = []
        for file in os.listdir(imgs):
            if not file.endswith(".jpg"):
                continue
            prefix = file.rsplit(".", 1)[0]
            if not os.path.exists(f"{labels}/{prefix}.txt"):
                print(f"{labels}/{prefix}.txt miss")
                continue
            tag.append(prefix)
        print(f"total images: {len(tag)}")
        random.shuffle(tag)
        print(f"shuffled, the first: {tag[0]}")
        train_size = int(args.proportion * len(tag))
        valid_size = len(tag) - train_size
        print(f"train:{train_size} valid:{valid_size}")
        move_data(PROJ,"train",tag[:train_size])
        move_data(PROJ,"valid",tag[train_size:])

if __name__ == "__main__":
    main()