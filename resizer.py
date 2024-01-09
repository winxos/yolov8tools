
'''
yolov8 raw image resize to (640,640)
winxos 20231214
### usage
directory structure

projs
-yolov8_resizer.py
-proj1
--raw

put your raw images (jpg) into dir ./proj1/raw/
run:
python yolov8_resizer.py --proj proj1

will make directory images in proj1
and all images resize to 640,640, keep ratio.
'''
import argparse
import os
import cv2
import numpy as np 
from tqdm import tqdm

def image_norm(img):
    old_size = img.shape[0:2]
    target_size = max(old_size[0],old_size[1])
    pad_w = target_size - old_size[1]
    pad_h = target_size - old_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0,0,0))
    return img_new
def main():
    parser = argparse.ArgumentParser(description="yolov8 image resize to (640,640),keep ratio")
    parser.add_argument("--proj",default="grains",help="project dir name")
    args = parser.parse_args()
    print(args)
    PROJ = args.proj
    imgs = f"{PROJ}/raw"
    os.makedirs(f"{PROJ}/images", exist_ok=True)
    for file in tqdm(os.listdir(imgs), desc=f"resizing", ascii=" =", colour="green"):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imdecode(np.fromfile(f"{imgs}/{file}",dtype=np.uint8),cv2.IMREAD_COLOR)
        norm = image_norm(img)
        norm = cv2.resize(norm,(640,640))
        cv2.imwrite(f"{PROJ}/images/{file}",norm)
    print("done.")
if __name__ == "__main__":
    main()