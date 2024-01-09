# yolov8 environment config

> wvv 20230212

### yolov8 install

```pip install ultralytics```

### usage
``` bash
#train
yolo detect train model=yolov8n.pt data=./hand/data.yaml epochs=300 imgsz=640 workers=4 batch=4
#segment
yolo segment train model=yolov8n-seg.pt data=./hand/data.yaml epochs=300 imgsz=640 workers=4 batch=16
#predict
yolo detect predict model=best.pt conf=0.25 source=2.jpg
yolo detect predict model=best.pt source=0 show=True
#track
yolo track model=best.pt source=0 show=True save=True
yolo classify train data=d:\GrainLibs\yolo-rice-raw model=yolov8n-cls.pt batch=512 translate=0 scale=0
```
### labeling
https://app.roboflow.com/ 在线标签工具，支持yolov8格式导出
x-anylabeling 本地标签工具

### training
自定义数据路径文件修改：
~\AppData\Roaming\Ultralytics\settings.yaml
修改： datasets_dir: C:\Users\admin\Desktop\ultralytics\

训练完在runs文件夹，根据输出内容可以找到训练好的模型

### export onnx
```shell
pip install onnx
yolo export model=seg.pt format=onnx opset=12
```

