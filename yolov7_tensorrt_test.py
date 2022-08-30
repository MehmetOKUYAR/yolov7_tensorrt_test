from unittest import result
import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import argparse


class yoloV7_tensorrt():
    def __init__(self,weights_path):
        w = weights_path
        device = torch.device('cuda:0')
        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):

            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()

        self.names = ["no_entry",
                        "trespass_sign",
                        "straight_or_left_only",
                        "straight_or_right_only",
                        "left_only",
                        "20_speed_limit_end",
                        "30_speed_limit",
                        "20_speed_limit",
                        "right_only",
                        "no_right_turn",
                        "no_left_turn",
                        "stop",
                        "no_parking",
                        "park",
                        "bus_stop",
                        "red_light",
                        "yellow_light",
                        "green_light", ]
        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.names)}


    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def postprocess(self,boxes,r,dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes

    

    def detect(self,img):
        device = torch.device('cuda:0')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)

        im = torch.from_numpy(im).to(device)
        im/=255
        start = time.perf_counter()
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data

        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]


        for box,score,cl in zip(boxes,scores,classes):
            box = self.postprocess(box,ratio,dwdh).round().int()
            name = self.names[cl]
            color = self.colors[name]
            name += ' ' + str(round(float(score),3))
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,8)
            cv2.putText(img,name,(int(box[0]), int(box[1]) - 15),cv2.FONT_HERSHEY_SIMPLEX,2,color,thickness=5,lineType=cv2.LINE_AA)

        return img



def main(args):
    weights_path = args["weights"]
    image_path = args["image"]
    video_path = args["video"]
    out_path = args["output"]
    yolov7 = yoloV7_tensorrt(weights_path)


    # ========== if input is image ==========
    if image_path is not None:
        image = cv2.imread(image_path)
        image = yolov7.detect(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if out_path is not None:
            result_name = "result.jpg"
            cv2.imwrite(out_path+result_name,image)
            
        cv2.imshow("image",cv2.resize(image,(1280,720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # =============== if input is video =====================
    elif video_path is not None:
        cap = cv2.VideoCapture(video_path)
        prev_frame_time=0
        new_frame_time=0

        # ========== if output is video save==========
        if out_path is not None:
            name = "result.mp4"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(width,height,fps)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path+name,fourcc,fps,(width,height))
        while True:
            ret, frame = cap.read()
            
            if ret:
                image = yolov7.detect(frame)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #======= video frame rate ======================
                new_frame_time=time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(image,"FPS : "+str(round(fps,2)),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,[0,0,255],thickness=5,lineType=cv2.LINE_AA)
                #================================================
            

                if out_path is not None:
                    out.write(image)
          
                cv2.imshow("image",cv2.resize(image,(1280,720)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", required=True, help="path to weights file")
    ap.add_argument("-i", "--image", help="path to input image")
    ap.add_argument("-v", "--video", help="path to input video file")
    ap.add_argument("-o", "--output", help="path to output video file")
    args = vars(ap.parse_args())
    main(args)

    
