# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.centroidtracker_advance import CentroidTracker
from utils.trackableobject import TrackableObject
import math
from datetime import datetime
from playsound import playsound
from multiprocessing import Process,Manager, Value

vehicle_move = Value("i", 0)

Wset=640
Hset=480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
now = datetime.now()
filename = now.strftime("./output/%d_%m_%Y_%H_%M_%S.mp4")
out = cv2.VideoWriter(filename,fourcc, 6.0, (Wset,Hset))
P1_BOTTOM_LEFT_X = 60
P1_BOTTOM_LEFT_Y = 480
P2_BOTTOM_RIGHT_X = 580
P2_BOTTOM_RIGHT_Y = P1_BOTTOM_LEFT_Y

P3_TOP_LEFT_X = 270
P3_TOP_LEFT_Y = 320
P3_TOP_RIGHT_X = 370
P3_TOP_RIGHT_Y = P3_TOP_LEFT_Y

(x1,y1) = (P1_BOTTOM_LEFT_X ,P1_BOTTOM_LEFT_Y)
(x2,y2) = (P2_BOTTOM_RIGHT_X,P2_BOTTOM_RIGHT_Y)

(x3,y3) = (P3_TOP_LEFT_X ,P3_TOP_LEFT_Y)
(x4,y4) = (P3_TOP_RIGHT_X,P3_TOP_RIGHT_Y)
def left_line(x,y):
    global x1,y1,x2,y2,x3,y3,x4,y4
    a = y1 - y3
    b = x3 - x1
    c = a*(x3) + b*(y3)
    return(a*x + b*y - c)
def right_line(x,y):
    global x1,y1,x2,y2,x3,y3,x4,y4
    a = y2 - y4
    b = x4 - x2
    c = a*(x2) + b*(y2)
    return(a*x + b*y - c)
def check_vehicle_in_lane(Px,Py,Qx,Qy):
    if ((left_line(Px+(Qx-Px)/4,Qy)*right_line(Px+(Qx-Px)/4,Qy) < 0) or (left_line(Qx-(Qx-Px)/4,Qy)*right_line(Qx-(Qx-Px)/4,Qy) < 0)):
        return 1
    else:
        return 0
def calc_distance(endY):
    if (Hset - endY>0):
        return (0.02852*(Hset - endY)+2.05872)
    else:
        return 0
def alert_monitor(vehicle_move):
	while(1):
		if (vehicle_move.value==1):
			playsound("/home/toanrd/Desktop/yolov5/audio/Vehicle_Department_short.mp3")
		else:
			time.sleep(0.1)

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    global x1,y1,x2,y2,x3,y3,x4,y4,vehicle_move
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    prevSpeed           = 0
    Speed               = 15
    viewRange           = 0
    threshDistance      = 0
    vehicleDepartCounter= 0
    listID              = []
    numberObjectDetect  = 0
    numberVehicleParking= 0
    numberVehicleDepart = 0
    VEHICLE_DEPARTMENT  = False
    COLLOSION_WARNING   = False
    STEERING_SIGNAL     = False
    REVERSE_SIGNAL      = False
    CollisionLeft       = False
    CollisionRight      = False
    frame               = 0
    for path, img, im0s, vid_cap in dataset:
        
        t1  = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            # Speed = cv2.getTrackbarPos("vehicle", windowName)
            if (prevSpeed != Speed):
                print("Speed: ",Speed)
            if (Speed >= 0) and (Speed <10):
                viewRange = 350
                threshDistance = 2.5
                (x1,y1) = (P1_BOTTOM_LEFT_X -60 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+60,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 10) and (Speed <20):
                viewRange = 340
                threshDistance = 3.5
                (x1,y1) = (P1_BOTTOM_LEFT_X -50 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+50,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 20) and (Speed <30):
                viewRange = 330
                threshDistance = 4.0
                (x1,y1) = (P1_BOTTOM_LEFT_X -40 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+40,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 30) and (Speed <40):
                viewRange = 320
                threshDistance = 5.0
                (x1,y1) = (P1_BOTTOM_LEFT_X -30 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+30,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 40) and (Speed <50):
                viewRange = 310
                threshDistance = 6.0
                (x1,y1) = (P1_BOTTOM_LEFT_X -15 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+15,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 50) and (Speed <60):
                viewRange = 300
                threshDistance = 7.0
                (x1,y1) = (P1_BOTTOM_LEFT_X -00 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X+00,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 60) and (Speed <70):
                viewRange = 285
                threshDistance = 8.0
                (x1,y1) = (P1_BOTTOM_LEFT_X +15 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X-15,P2_BOTTOM_RIGHT_Y)
            elif (Speed >= 70):
                viewRange = 270
                threshDistance = 9.0
                (x1,y1) = (P1_BOTTOM_LEFT_X +30 ,P1_BOTTOM_LEFT_Y)
                (x2,y2) = (P2_BOTTOM_RIGHT_X-30,P2_BOTTOM_RIGHT_Y)

            if (STEERING_SIGNAL == True):
                VEHICLE_DEPARTMENT = False
                COLLOSION_WARNING = False
            elif (STEERING_SIGNAL == False):
                if (Speed <= 5):
                    VEHICLE_DEPARTMENT = True
                    COLLOSION_WARNING  = False
                elif (Speed > 5):
                    VEHICLE_DEPARTMENT = False
                    COLLOSION_WARNING  = True


            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            # im0 = cv2.flip(im0,0)
            # im0 = cv2.flip(im0,1)
            im0 = cv2.resize(im0, (Wset,Hset), interpolation = cv2.INTER_AREA)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)

            # Visualize
            if (VEHICLE_DEPARTMENT is True):
                cv2.putText(im0, "Vehicle Department", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            else:
                vehicleDepartCounter = 0
                vehicle_move.value = 0
                cv2.putText(im0, "Vehicle Department", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
            if (COLLOSION_WARNING is True):
                cv2.putText(im0, "Collision Warning", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            else:
                cv2.putText(im0, "Collision Warning", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
            cv2.putText(im0, "Left/Right Sign", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
            cv2.putText(im0, "Reverse", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                LANE_OVERRIDE = 0

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')

                    # annotator.box_label(xyxy, label, color=colors(c, True))
                    startX = int(xyxy[0].item())
                    startY = int(xyxy[1].item())
                    endX   = int(xyxy[2].item())
                    endY   = int(xyxy[3].item())
                    # check vehicle is on lane or not
                    if (check_vehicle_in_lane(startX,startY,endX,endY)==1) and (endY > viewRange):
                        LANE_OVERRIDE = LANE_OVERRIDE + 1
                        
                        if (VEHICLE_DEPARTMENT is True):
                            if ((left_line(startX+(endX-startX)/4,endY)*right_line(startX+(endX-startX)/4,endY) < 0) and (left_line(endX-(endX-startX)/4,endY)*right_line(endX-(endX-startX)/4,endY) < 0) and (abs(endX+startX-Wset) < 100)):
                                if (label == "car") or (label == "truck") or (label == "bus"):
                                    cv2.rectangle(im0,(startX, startY),(endX, endY),(125,125,250), 2)
                                    numberObjectDetect = numberObjectDetect + 1
                                elif (label == "person") or (label == "motorcycle") or (label == "bicycle"):
                                    if (left_line(endX,endY)*right_line(endX,endY) < 0) and (left_line(startX,endY)*right_line(startX,endY) < 0):
                                        if (((startX+endX)-Wset) < (endX-startX)) and (endY > 320):
                                            cv2.rectangle(im0,(startX, startY),(endX, endY),(125,125,250), 2)
                                            numberObjectDetect = numberObjectDetect + 1
                                if (calc_distance(endY)) < 4:
                                    numberVehicleParking = numberVehicleParking + 1
                                elif (calc_distance(endY)) >= 4:
                                    numberVehicleDepart = numberVehicleDepart + 1
                                dist = calc_distance(endY)
                                info = [
                                        ("ID  ", '{}'.format(label)),
                                        ("Dist", '{:.1f}'.format(dist)),
                                    ]
                                for (l, (k, v)) in enumerate(info):
                                    text = "{}:{}".format(k, v)
                                    cv2.rectangle(im0,(startX, startY + l*15),(startX+len(text)*7, startY + l*15+15),(0,125,250), -1)
                                    cv2.putText(im0, text, (startX,startY + l*15+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)            
                        if (COLLOSION_WARNING is True):
                            if ((label == "car") or (label == "truck") or (label == "bus") or (label == "person") or (label == "motorcycle") or (label == "bicycle")):
                                numberObjectDetect = numberObjectDetect + 1
                                Cx = (int)((startX + endX)/2)
                                Cy = endY
                                dist = calc_distance(endY)
                                
                                info = [
                                        ("ID  ", '{}'.format(label)),
                                        ("Dist", '{:.1f}'.format(dist)),
                                    ]
                                if (dist < threshDistance):
                                    cv2.putText(im0, "Collision Warning", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
                                    if (abs(left_line((startX+endX)/2,endY)) < abs(right_line((startX+endX)/2,endY))):
                                        CollisionLeft = True

                                    if ( abs(left_line((startX+endX)/2,endY)) > abs(right_line((startX+endX)/2,endY)) ):
                                        CollisionRight = True

                                    cv2.rectangle(im0,(startX, startY),(endX, endY),(0,0,250), 4)
                                else:
                                    cv2.rectangle(im0,(startX, startY),(endX, endY),(125,125,250), 2)

                                for (l, (k, v)) in enumerate(info):
                                    text = "{}:{}".format(k, v)
                                    cv2.rectangle(im0,(startX, startY + l*15),(startX+len(text)*7, startY + l*15+15),(0,125,250), -1)
                                    cv2.putText(im0, text, (startX,startY + l*15+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)                            

                # Visualize
                if (LANE_OVERRIDE > 0):
                    cv2.line(im0, (x1,y1), (x3,y3), (25,125, 255), 2)
                    cv2.line(im0, (x2,y2), (x4,y4), (25,125, 255), 2)
                elif (LANE_OVERRIDE == 0):
                    cv2.line(im0, (x1,y1), (x3,y3), (255, 125, 0), 2)
                    cv2.line(im0, (x2,y2), (x4,y4), (255, 125, 0), 2)
                if (CollisionLeft is True):
                    cv2.line(im0, (x1,y1), (x3,y3), (0,0, 255), 4)
                if (CollisionRight is True):
                    cv2.line(im0, (x2,y2), (x4,y4), (0,0, 255), 4)
                if (VEHICLE_DEPARTMENT is True):
                    if (numberVehicleParking == 0) and (numberVehicleDepart > 0):
                        vehicleDepartCounter = vehicleDepartCounter + 1
                        if (vehicleDepartCounter >= 2):
                            cv2.putText(im0, "Vehicle Department", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
                            vehicle_move.value = 1
                    else:
                        vehicleDepartCounter = 0
                        vehicle_move.value = 0
            else:
                cv2.line(im0, (x1,y1), (x3,y3), (255, 125, 0), 2)
                cv2.line(im0, (x2,y2), (x4,y4), (255, 125, 0), 2)
            CollisionLeft = False
            CollisionRight = False
            numberVehicleDepart = 0
            numberVehicleParking = 0
            t2 = time_sync()
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            with open("./log.txt", "a+") as file_object:
                file_object.write(f'FPS,{frame},{1/(t2 - t1):.1f}\n')
            frame = frame + 1
            if (1/(t2 - t1)>=4):
                cv2.putText(im0, f"FPS: {1/(t2 - t1):.1f}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            elif (1/(t2 - t1)<4):
                cv2.putText(im0, f"FPS: {1/(t2 - t1):.1f}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,255), 1)
            # Stream results
            # im0 = annotator.result()
            if (Speed < 20):
                cv2.putText(im0, "Speed: {}".format(Speed), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)
            elif (Speed >= 20)and((Speed < 50)):
                cv2.putText(im0, "Speed: {}".format(Speed), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,250), 1)
            elif (Speed >= 50):
                cv2.putText(im0, "Speed: {}".format(Speed), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,250), 2)
            prevSpeed = Speed
            if (STEERING_SIGNAL is True):
                cv2.putText(im0, "Left/Right Signal", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,255), 2)
            if (REVERSE_SIGNAL is True):
                cv2.putText(im0, "Reverse", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,125,255), 2)
            # Save results (image with detections)
            print(im0.shape[:2])
            if True:
                out.write(im0)

            if True:
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)  # 1 millisecond
                if key == ord("q"):
                    break
                elif key == ord("w"):
                    Speed = Speed + 5
                elif key == ord("s"):
                    Speed = Speed - 5
                    if (Speed < 0):
                        Speed = 0
                elif (key == ord("d")) or (key == ord("a")):
                    STEERING_SIGNAL = True
                elif key == ord("x"):
                    STEERING_SIGNAL = False 
                    REVERSE_SIGNAL = False
                elif key == ord("c"):
                    cv2.imwrite("single_image.jpg", im0)
                elif key == ord("z"):
                    REVERSE_SIGNAL = True


    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt
def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

thread_1 = Process(target=alert_monitor,args=(vehicle_move,))
thread_1.start()
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

