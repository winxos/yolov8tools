### yolov8tools

winxos 20231214

#### this repo contain all the tools you need for beginner using yolov8

* **yolov8_resizer.py**  [resize the image to 640,640]

* **yolov8_label_converter.py** [convert json label to yolov8 txt format]
* **yolov8_spliter.py** [random split the dataset to train and valid]

# the step using yolov8

1. capture some images
2. preprocessing the images to fit yolov8 need
3. labeling the image, using anylabeling, or other software
4. convert the label to yolov8 format
5. train the model
6. test the model

### details using this repo

create proj1 in the same path with yolov8_resizer.py

put the images into folder raw in the proj1



```
repo
-yolov8_resizer.py
-proj1
--raw
```

config the yolov8 environment

run command below in the shell.

```bash
python yolov8_resizer.py --proj proj1
```

then will create images folder in the proj1, and the images resized to (640,640) for yolov8

the using anylabeling or other labeling tool to labeling the images.

then you will get json file in the images folder.

run

```bash
python yolov8_label_converter.py --proj proj1
```

then you will get labels in txt format in folder proj1/labels/

you can found some other args setting in the yolov8_label_converter.py

like labeling type [box or polygon]

then you can using **yolov8_spliter.py** to split the dataset to train and valid

run

```bash
python yolov8_spliter.py --proj proj1
```

the proportion can set using --proportion argument

then you will get train and valid folder dataset for yolov8 training.

you need to create data.yaml file to config the classess info.

```yaml
train: ../train/images
val: ../valid/images

nc: 10
names: ['1', '2', '3', '4', '5', '6', '7', '8', '9','10']
```

nc is the count of label

you can using cli to train the model.

```bash
yolo detect train model=yolov8n.pt data=./proj1/data.yaml epochs=300 imgsz=640 workers=4 batch=4
```

or predict the model

```bash
yolo detect predict model=best.pt source=0 show=True
```

source=0 means using the camera for input



