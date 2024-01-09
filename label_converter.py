'''
yolov8 label converter
init from x-anylabeling/tools/label_convert.py
winxos 20231213

### usage
this script can easily
convert anylabeling json to yolov8 txt, for yolo train
or convert yolov8 txt to anylabeling json, for labeling
the data project directory:

projs/
-yolov8_label_converter.py
-proj1/
--images/
--labels/
--data.yaml
-proj2/
--images/
--labels/
--data.yaml

data.yaml is the classes info file for yolo training, format like:

train: ../train/images
val: ../valid/images

nc: 2
names: ['aa', 'bb']

'''
import argparse
import json
import os
import os.path as osp
import time
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys
import yaml

sys.path.append(".")

VERSION = "1.0.0"


class Converter:
    def __init__(self, classes_data=None):
        self.classes = classes_data
        print(f"classes is: {self.classes}")

    def reset(self):
        self.custom_data = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1,
        )

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height




class RectLabelConverter(Converter):
    def json_to_yolov8(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]

                class_index = self.classes.index(label)

                x_center = (points[0][0] + points[1][0]) / (2 * image_width)
                y_center = (points[0][1] + points[1][1]) / (2 * image_height)
                width = abs(points[1][0] - points[0][0]) / image_width
                height = abs(points[1][1] - points[0][1]) / image_height

                f.write(
                    f"{class_index} {x_center} {y_center} {width} {height}\n"
                )

    def yolov8_to_json(self, input_file, output_file, image_file):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)

        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            cx = float(line[1])
            cy = float(line[2])
            nw = float(line[3])
            nh = float(line[4])
            xmin = int((cx - nw / 2) * img_w)
            ymin = int((cy - nh / 2) * img_h)
            xmax = int((cx + nw / 2) * img_w)
            ymax = int((cy + nh / 2) * img_h)

            shape_type = "rectangle"
            label = self.classes[class_index]
            points = [
                [xmin, ymin],
                [xmax, ymax]
            ]
            shape = {
                "label": label,
                "text": "",
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            }
            self.custom_data["shapes"].append(shape)
        self.custom_data["imagePath"] = os.path.basename(image_file)
        self.custom_data["imageHeight"] = img_h
        self.custom_data["imageWidth"] = img_w
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

   

class PolyLabelConvert(Converter):

    def json_to_yolov8(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        image_size = np.array([[image_width, image_height]])

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                label = shape["label"]
                points = np.array(shape["points"])
                class_index = self.classes.index(label)
                norm_points = points / image_size
                f.write(
                    f"{class_index} "
                    + " ".join(
                        [
                            " ".join([str(cell[0]), str(cell[1])])
                            for cell in norm_points.tolist()
                        ]
                    )
                    + "\n"
                )

    def yolov8_to_json(self, input_file, output_file, image_file):
        self.reset()

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        image_width, image_height = self.get_image_size(image_file)
        image_size = np.array([image_width, image_height], np.float64)

        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = self.classes[class_index]
            masks = line[1:]
            shape = {
                "label": label,
                "text":"",
                "points": [],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            for x, y in zip(masks[0::2], masks[1::2]):
                point = [np.float64(x), np.float64(y)]
                point = np.array(point, np.float64) * image_size
                shape["points"].append(point.tolist())
            self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

def main():
    valid_modes = ["json2yolo", "yolo2json"]
    parser = argparse.ArgumentParser(description="yolov8 converter")
    parser.add_argument("--proj",default="grains",help="project dir name")
    parser.add_argument("--type",default="polygon",choices=["box", "polygon"])
    parser.add_argument("--mode", default="json2yolo",  choices=valid_modes)
    args = parser.parse_args()
    PROJ = args.proj
    args.img_path = f"{PROJ}/images"
    args.label_path = f"{PROJ}/labels"
    with open(f"{PROJ}/data.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("data.yaml: ",cfg)
    print(f"{args.mode} {args.type}")
    start_time = time.time()
    
    if args.type == "box":
        converter = RectLabelConverter(cfg["names"])
        assert (args.mode in valid_modes), f"type only supported in {valid_modes}"
    elif args.type == "polygon":
        converter = PolyLabelConvert(cfg["names"])
        assert (args.mode in valid_modes), f"type only supported in {valid_modes}"

    if args.mode == "json2yolo":
        file_list = os.listdir(args.img_path)
        os.makedirs(args.label_path, exist_ok=True)
        for file_name in tqdm(file_list, desc="converting", unit="file", colour="green"):
            if not file_name.endswith(".json"):
                continue
            src_file = osp.join(args.img_path, file_name)
            dst_file = osp.join(args.label_path, osp.splitext(file_name)[0] + ".txt" )
            converter.json_to_yolov8(src_file, dst_file)
    elif args.mode == "yolo2json":
        img_dic = {}
        for file in os.listdir(args.img_path):
            if not file.endswith(".jpg"):
                continue
            prefix = file.rsplit(".", 1)[0]
            img_dic[prefix] = file
        file_list = os.listdir(args.label_path)
        for file_name in tqdm(file_list, desc="converting", unit="file", colour="green"):
            src_file = osp.join(args.label_path, file_name)
            dst_file = osp.join(args.img_path, osp.splitext(file_name)[0] + ".json")
            img_file = osp.join(
                args.img_path, img_dic[osp.splitext(file_name)[0]]
            )
            converter.yolov8_to_json(src_file, dst_file, img_file)
   
    end_time = time.time()
    print(f"done. time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
