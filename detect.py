import torch
from net import UNet
import torchvision  # 数据预处理
from PIL import Image  # 图片处理
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import math

FILE = Path(__file__).resolve()  # 获取文件绝对路径，并解析如何符号链接或相对文件路径以获取实际文件路径
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取当前脚本的直接父目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

suma = 0  # 定义一个全局变量
output = []  # 定义一个空列表用于存储检测结果

"""
run函数运行目标检测模型，并根据输入参数进行初始化设置
接下来，函数根据输入的数据源（图像文件、目录或摄像头）创建数据加载器，用于加载输入数据。
函数开始运行目标检测算法，并对每个图像进行处理。它通过模型进行推断，得到目标的预测结果。
预测结果经过非极大值抑制（NMS）处理，去除重叠较多的检测框。
函数根据预测结果将目标框和标签添加到图像上，并将结果保存到指定的文件或显示在屏幕上。
最后，函数会输出检测速度和保存的结果数量。
"""


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold  置信度阈值
        iou_thres=0.45,  # NMS IOU threshold  iou阈值
        max_det=1000,  # maximum detections per image  每个图像的最大检测
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_dir=None, txt_path=None, save_path=None):
    global suma, output
    output = []
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans

    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    if pt:

        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            if "edgetpu" in w:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                import tflite_runtime.interpreter as tflri
                delegate = {'Linux': 'libedgetpu.so.1',  # install libedgetpu https://coral.ai/software/#edgetpu-runtime
                            'Darwin': 'libedgetpu.1.dylib',
                            'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = tflri.Interpreter(model_path=w, experimental_delegates=[tflri.load_delegate(delegate)])
            else:
                interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0

    for path, img, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            # print(img.shape)
            # print(type(img))
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # print(det)
                suma = 0
                ################################################################################
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        suma += c + 1
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        output.append(([int(i) for i in xyxy], c))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                print(suma)
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            # 毛囊目标检测预测结果显示
            if True:
                cv2.imshow(str(p), im0)  # 1 millisecond

            # Save results (image with detections)
            if not save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    # 使用argparse模块解析命令行参数
    parser = argparse.ArgumentParser()
    # 指定模型权重文件路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'tphyolov5.pt', help='model path(s)')  #
    # 指定推理的输入源，可以是文件，目录，URL或网络摄像头
    parser.add_argument('--source', type=str, default=ROOT / './output', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    # 指定推理的输入图像大小，接受两个参数：高度和宽度，默认值为【640】
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # 指定检测的置信度阈值，默认值为0.25
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 指定非极大抑制的iou(交并比)阈值，默认为0.45
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 指定每张图片的最大检测数，默认值为1000.
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 指定推理所使用的设备，如GPU或CPU，默认值为''
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 标志，指示是否显示推理结果，
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 标志，指示是否将文件保存到文本文件中，
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 标志，指示是否在文本文件中保存置信度分数，
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 标志，指示是否保存裁剪后的预测框
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # 标志，指示是否保存图像/视频
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 指定要筛选的检测类别，可以是一个或多个整数参数
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 标志，指示是否使用类别不可知的非极大抑制，
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 标志，指示是否使用增强推理，
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 标志，指示是否可视化特征，
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # 标志，指示是否更新所用模型
    parser.add_argument('--update', action='store_true', help='update all models')
    # 指定保存结果的目录，
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 指定保存结果的目录
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 标志，指示同名项目已存在是否可行，不需要增加编号
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 指定边界框的线条粗细(像素)，默认值为3
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 标志，指示是否隐藏标签，
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 标志，指示是否隐藏置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 标志，指示是否使用半精度(FP16)推理，
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 标志，指示是否使用opencv dnn进行ONNX推理，
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand 扩展opt.imgsz列表
    print_args(FILE.stem, opt)  # 打印解析后的参数
    return opt


totensor = torchvision.transforms.ToTensor()

namee = "190"


# 语义分割
def uet_detect():
    """
    创建一个UNet模型，并将其移动到GPU上。
    定义图像预处理步骤，包括将图像调整为固定大小（256x256）。
    加载图像，并根据最长边生成一个全零的正方形矩阵。
    将原始图像粘贴到正方形矩阵中央，以保持图像的长宽比。
    对图像进行透明度处理，将其转换为RGBA模式，并将每个通道拆分为独立的通道。
    将图像转换为torch.Tensor格式，并将其形状调整为(1, 3, 256, 256)。
    加载预训练的UNet模型，并将其设置为评估模式。
    对输入图像进行预测，并得到预测输出。
    将预测输出转换回numpy数组格式，并进行一些后处理操作。
    计算预测输出中非零像素的数量（代表分割的区域面积）。
    返回处理后的图像、分割区域的面积以及原始图像的尺寸。
"""
    net = UNet().cuda()  # 创建一个UNet模型，并将其移到GPU上，
    # 图像预处理部分
    resizer = torchvision.transforms.Resize((256, 256))  # 调整图像大小为(256,256)
    img_data = f"./{namee}.jpg"  #
    img = Image.open(img_data)

    w, h = img.size
    www, hhh = img.size
    # 以最长边为基准，生成全0正方形矩阵
    slide = max(h, w)
    black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
    black_img.paste(img, (0, 0, int(w), int(h)))  # patse在图中央和在左上角是一样的
    img = resizer(black_img)

    # 透明度
    imgt = img.convert("RGBA")
    r, g, b, alpha = imgt.split()
    imgt = np.array(imgt)
    imgtt = np.array(imgt)
    img = totensor(img)

    # 原图
    # plt.subplot(1, 2, 1)
    # plt.imshow(img.numpy().transpose(1, 2, 0))

    img = torch.reshape(img, (1, 3, 256, 256))  # 将图像转换为torch.Tensor格式，并将其形状调整为(1, 3, 256, 256)
    img = img.cuda()

    # 加载模型
    net.load_state_dict(torch.load('D:/cfd/SAVE/Unet40.pt'))
    net.eval()
    # 预测
    img_out = net(img)

    img_out = img_out[0].cpu().detach()

    nout = img_out.numpy().transpose(1, 2, 0)
    area = 0
    for wn, w in enumerate(nout):
        for hn, h in enumerate(w):
            # print(h.sum())
            if h.sum() > 1e-3:
                area += 1
            else:
                imgt[wn][hn][3] = 0

    return imgt, area, (www, hhh)  # 返回处理后的图像，分割区域的面积以及原始图像的尺寸


imgt, area, imgsize = uet_detect()  # 调用uet_detect()函数，获得语义分割的图像imgt，分割区域的面积area和原始图像尺寸imgsize

# 相机参数


import exifread

try:
    f = open(f"./{namee}.jpg", 'rb')
    contents = exifread.process_file(f)
    f.close()
    # print(contents)
    efl = int(str(contents['EXIF FocalLengthIn35mmFilm']))
except:
    efl = 24  # 读取图像EXIF数据，提取等效焦距efl作为相机参数
# print(efl)

# efl = int(efi.values)
# print(efi)
# efl = 24  # 等效焦距，单位为毫米
# 每个像素在物理尺寸上的大小（假设相机为1/2.3英寸传感器）根据相机传感器的尺寸和图像的尺寸比例

sensor_width = 6.17  # 单位为毫米
sensor_height = 4.55  # 单位为毫米

pixel_width = sensor_width / (imgsize[0])
pixel_height = sensor_height / (imgsize[1])
# 计算焦距和光圈大小
focal_length = efl / 1000


# 计算每个轮廓的面积，并将其转换为单位面积（平方厘米）

def scaale(area):  # 定义scaale函数，用于计算每个轮廓的面积，并将其转换为单位面积(平方厘米)
    pixel_num = area  # 像素数量
    scaled_pixel_num = pixel_num * (256 / imgsize[0]) * (256 / imgsize[1])
    # 计算缩放后的面积并转换为单位面积（平方厘米）
    scaled_area_in_pixel = scaled_pixel_num * pixel_width * pixel_height  # 像素面积（物理尺寸）
    scaled_area_in_mm = scaled_area_in_pixel / (focal_length ** 2)  # 平方毫米
    scaled_area_in_cm = scaled_area_in_mm / 100  # 平方厘米
    return scaled_area_in_cm, scaled_area_in_mm


# 显示并保存处理后的图像
plt.figure()
plt.imshow(imgt)
plt.imsave("./output/0.jpg", imgt)
plt.close()


def main(opt):  # 定义main()函数，其中调用了parse_opt()函数和run函数(),用于解析命令行参数并运行相关操作
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


matplotlib.use('TkAgg')
plt.figure()
orinn = Image.open(f"./{namee}.jpg")
plt.axis('off')
plt.subplot(1, 2, 1)
plt.imshow(orinn)
plt.subplot(1, 2, 2)
plt.imshow(imgt)

opt = parse_opt()
main(opt)

cm, mm = scaale(area)  # 调用scaale()函数计算分割区域面积，并将其转换为单位面积(平方厘米)和单位面积(平方毫米)
longofhair = []
maolang = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for ni, i in enumerate(output):
    # plt.figure()
    # plt.subplot(1, 2, 1)
    imgtt = Image.fromarray(imgt)
    # print(i[0][0], i[0][1], i[0][2], i[0][3])
    imgtt = imgtt.crop((i[0][0], i[0][1], i[0][2], i[0][3]))
    nimgtt = np.array(imgtt)
    areaa = 0
    for xgtn, xgt in enumerate(nimgtt):
        for ygtn, ygt in enumerate(xgt):
            if ygt[:-2].sum() > 200:
                nimgtt[xgtn][ygtn][3] = 0
                areaa += 1
    # print(areaa)
    lee = scaale(areaa / 8)[1] ** 0.5
    # print(lee / 10)
    if lee:
        longofhair.append(lee / 10)  # 对输出结果进行进一步处理，计算每个区域的长度，并将其存储在longofhair列表中
    else:
        pass
    # plt.imshow(nimgtt)
    # plt.show()
    maolang[i[1]] += 1
# print(maolang.values())
hairrate = [i / len(output) for i in maolang.values()]
# print(longofhair)
xi, zhong, chu = 0, 0, 0  # 统计不同类型的毛发数量，并计算各类型毛发的比例
for i in longofhair:
    if i > 0.05:
        chu += 1
    elif 0.03 < i < 0.05:
        zhong += 1
    else:
        xi += 1
# 统计毛囊密度，平均直径和毛囊数量比例等数据，并输出结果
xirate = xi / len(longofhair)
zhongrate = zhong / len(longofhair)
churate = chu / len(longofhair)
print(f"细发(<0.03mm)比例:\t\t\t{xirate}\n中间发(0.03mm~0.05mm)比例:\t{zhongrate}\n粗发(>0.05mm)比例:\t\t\t{churate}")
print(
    f'大于等于3根毛发毛囊比例：\t{1 - hairrate[0] - hairrate[1]}\n2根毛发的毛囊比例:\t\t\t{hairrate[1]}\n单根毛发的毛囊比例：\t\t{hairrate[0]}')
print('平均直径:\t', np.mean(longofhair))
print('毛囊密度：\t', suma / cm)
if True:  # 如果输出结果为True，则显示图像
    plt.show()
    cv2.waitKey(0)
