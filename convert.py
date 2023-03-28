import os
import cv2
import numpy as np
import json
from shutil import move
import argparse

class ConverAnnotation:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        self.class_names = {"0": "person", "1":"car"}

    def get_class_id(self, class_name):
        for key, val in self.class_names.items():
            if val == class_name:
                return key
    
    def xywh_to_xyxy(self, xywh, img_w = None, img_h = None, dtype = None):
        x1 = xywh[0] - xywh[2]/2
        y1 = xywh[1] - xywh[3]/2
        x2 = xywh[0] + xywh[2]/2
        y2 = xywh[1] + xywh[3]/2
        if img_h is None or img_w is None:
            return [x1,y1,x2,y2]
        else:
            xyxy = [x1*img_w,y1*img_h,x2*img_w,y2*img_h]
            xyxy = np.array(xyxy, dtype=dtype) if dtype is not None else  np.array(xyxy, dtype=float)
            return xyxy
        
    def xyxy_to_xywh(self, xyxy, img_w = None, img_h = None):
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        x = xyxy[0] + w /2
        y = xyxy[1] + h/2
        if img_h is None or img_w is None:
            return [x,y,w,h]
        else:
            xywh = [x/img_w, y/img_h, w/img_w,h/img_h]
            return xywh
        
        
    def yolo_to_labelme(self):
        for fn in os.listdir(self.data_folder):
            if fn.endswith(".txt"):
                img_fp = os.path.join (self.data_folder, fn.replace("txt", "jpg"))
                if not os.path.exists(img_fp):
                    continue
                image = cv2.imread(img_fp)
                img_h, img_w,_ = image.shape

                yolo_gt = open(os.path.join(self.data_folder, fn))
                meta_data = {}
                meta_data["version"] = "5.1.1"
                meta_data["flags"] = {}
                meta_data["imagePath"] = fn.replace("txt", "jpg")
                meta_data["imageData"] = None
                meta_data["imageHeight"] = img_h
                meta_data["imageWidth"] = img_w

                shapes = []
                for line in yolo_gt:
                    line = line.split("\n")[0].split(" ")
                    label = self.class_names[line[0]]
                    xywh = np.array([line[1], line[2], line[3], line[4]], dtype=float)
                    xyxy = self.xywh_to_xyxy(xywh, img_w, img_h)
                    points = [[xyxy[0], xyxy[1]], [xyxy[2], xyxy[3]]]

                    shape = {"label": label, 
                             "points": points, 
                             "group_id": None, 
                             "shape_type": "rectangle", 
                             "flags": {}}
                    shapes.append(shape)
                meta_data["shapes"] = shapes

                json_file = os.path.join(self.data_folder, fn.replace("txt","json"))
                with open(json_file, 'w') as fp:
                    json.dump(meta_data, fp)
                
    def labelme_to_yolo(self, remove_prelabel):
        for fn in os.listdir(self.data_folder):
            if fn.endswith(".json"):
                json_fp = os.path.join(self.data_folder, fn)

                with open(json_fp) as json_file:
                    data = json.load(json_file)
                img_w = data["imageWidth"]
                img_h = data["imageHeight"]
                shapes = data["shapes"]
                line = ""
                for shape in shapes:
                    class_id = self.get_class_id(shape["label"])
                    points = shape["points"]
                    points = [points[0][0], points[0][1],points[1][0], points[1][1]]
                    xywh = self.xyxy_to_xywh(points,img_w, img_h)
                    line += f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"
                
                pre_label_fp = os.path.join(self.data_folder, fn.replace("json", "txt"))
                if remove_prelabel:
                    os.remove(pre_label_fp)
                else:
                    move(pre_label_fp, pre_label_fp.replace(".txt", "_pre.txt"))

                new_label = open(os.path.join(self.data_folder, fn.replace("json", "txt")),"a")
                new_label.write(line)
                new_label.close()
    
    def clean(self,json=None, pre_label=None):
        for fn in os.listdir(self.data_folder):
            fp = os.path.join(self.data_folder, fn)
            if fn.endswith(".json") and json is not None:
                os.remove(fp)
            if "_pre" in fn and json is not None:
                os.remove(fp)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data", help='data directory path')
    parser.add_argument('--type', default=1, help='convertion type')  
    parser.add_argument('--clean', default=True, help='clean all tmp files')   
    parser = parser.parse_args() 

    converter = ConverAnnotation(parser.data) 
    if parser.type == 0:
        converter.yolo_to_labelme()
    elif parser.type == 1:
        converter.labelme_to_yolo(remove_prelabel=False)
        if parser.clean:
            converter.clean(json=True, pre_label=True) 