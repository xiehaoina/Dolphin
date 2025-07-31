from doclayout_yolo import YOLOv10
from typing import List
from PIL import Image
from typing import Any

class DocLayoutYOLOModel(object):
    def __init__(self, config: Any):
        weight = config.weight
        self.device = config.device
        self.model = YOLOv10(weight)


    def inference(self, image:Image.Image):
        layout_res = []
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280,
            conf=0.10,
            iou=0.45,
            verbose=False, device=self.device
        )[0]
        for xyxy, conf, cla in zip(
            doclayout_yolo_res.boxes.xyxy.cpu(),
            doclayout_yolo_res.boxes.conf.cpu(),
            doclayout_yolo_res.boxes.cls.cpu(),
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            }
            layout_res.append(new_item)
        return layout_res

    def batch_inference(self, images: List[Image.Image]): 
        doclayout_yolo_res = [
            image_res.cpu()
            for image_res in self.model.predict(
                images,
                imgsz=1280,
                conf=0.10,
                iou=0.45,
                verbose=False,
                device=self.device,
            )
        ]
        layout_res = []
        for image_res in doclayout_yolo_res:
            for xyxy, conf, cla in zip(
                image_res.boxes.xyxy,
                image_res.boxes.conf,
                image_res.boxes.cls,
            ):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 3),
                }
                layout_res.append(new_item)
        return layout_res
