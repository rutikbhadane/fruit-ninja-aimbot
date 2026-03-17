import traceback

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.train(data='c:/Fruit-Ninja/config/dataset.yaml', epochs=1, imgsz=320)
except Exception as e:
    with open('c:/Fruit-Ninja/yolo_trace.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(traceback.format_exception(None, e, e.__traceback__)))
