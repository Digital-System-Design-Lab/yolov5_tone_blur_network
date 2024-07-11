# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn

import pandas as pd
import torch
#import torchvision.transforms.functional as B

#-----------bpp---------------------
import torchjpeg.torchjpeg.src.torchjpeg.dct as TJ_dct#torchjpeg 에서 dct 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.dct._color as TJ_ycbcr #추가
import torchjpeg.torchjpeg.src.torchjpeg.quantization.ijg as TJ_ijg #torchjpeg에서 quantization 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.dct._block as TJ_block #추가
from einops import rearrange # pip install einops (텐서를 좀 더 자유자재로 쓸수있게함)
def gaussian_blur(image, kernel_size, sigma):
    """
    이미지에 가우시안 블러를 적용합니다.
    image: 이미지 데이터를 나타내는 텐서
    kernel_size: 커널의 크기 (홀수)
    sigma: 가우시안 분포의 표준편차
    """
    # 가우시안 커널 생성
    image = image.half()
    sigma = sigma.to(image.device).half()
    radius = kernel_size // 2
    kernel_size = [kernel_size, kernel_size]
    x_coords = torch.arange(kernel_size[0]).float() - radius
    x_grid = x_coords.repeat(kernel_size[1]).view(kernel_size[1], kernel_size[0])
    y_grid = x_grid.t()
    sq_dist = x_grid ** 2 + y_grid ** 2
    sq_dist = sq_dist.to(image.device)
    kernel = torch.exp(-sq_dist / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # 커널을 이미지의 차원과 맞춰주기
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(image.size(1), 1, 1, 1)  # 이미지 채널 수에 맞춰 커널을 반복합니다
    kernel = kernel.half()
    # 이미지에 블러 적용
    padding = radius
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=image.size(1))
    return blurred_image

def delta_encode(coefs):#DCT 계수에 대해 / 델타 인코딩 수행(데이터 압축에서 자주 사용되는 기법)
    #델타 인코딩은 연속된 데이터 사이의 차이(델타)만 저장하는 방식
    #입력은 DCT 계수 텐서(coefs) // 각 블록의 dc 계수에 대해 델타 인코딩 적용 / ac는 그대로 유지
    #coefs 크기 (B, C, H*W/64, 64)
    #H*W/64가 블록갯수임
    ac = coefs[..., 1:]             # b 1 4096 63 #나머지는 AC 계수(63개)
    dc = coefs[..., 0:1]            # b 1 4096 1 #첫번째 요소는 DC계수(1개) 모든 블록에서 DC값을 추출
    dc = torch.cat([dc[..., 0:1, :], dc[..., 1:, :] - dc[..., :-1, :]], dim=-2)#각 DC 계수에서 바로 이전 DC 계수를 빼는 연산 
    #첫번째 DC 계수를 그대로 두고(델타 인코딩에서 시작점으로 사용), 그 이후 각 DC 계수에서 바로 이전 DC 계수를 뱨는 연산을 수행
    return torch.cat([dc, ac], dim=-1) # 델타 인코딩된 계수를 반환(데이터 중복성을 줄이고 압축률을 개선하는데 도움)

class DC_Predictor(nn.Module):# DC 값은 1개(0,0 point)
    def __init__(self):
        super(DC_Predictor, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.reshape(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AC_Predictor(nn.Module):# AC 값은 63개 (0,0 제외) 8x8 block 기준
    def __init__(self):
        super(AC_Predictor, self).__init__()
        self.lstm = nn.LSTM(1, 16, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(32 * 63, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.reshape(-1, 63, 1)
        x, _ = self.lstm(x)
        x = x.reshape(-1, 32 * 63)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Model_bpp_estimator(nn.Module):#bpp 추정
    def __init__(self):
        super(Model_bpp_estimator, self).__init__()
        self.dc_predictor = DC_Predictor()
        self.ac_predictor = AC_Predictor()

    def forward(self, x):
        dc_cl = self.dc_predictor(x[..., 0])
        ac_cl = self.ac_predictor(x[..., 1:])
        outputs = dc_cl + ac_cl
        return outputs
    
class ReinhardToneMapping(nn.Module):#HDR to LDR 
    def __init__(self, white_point=1.0):
        super(ReinhardToneMapping, self).__init__()
        self.white_point = white_point

    def forward(self, hdr_image, predicted_weights):
        """
        Apply the Reinhard tone mapping operator to an HDR image using dynamic luminance weights.
        
        Parameters:
        - hdr_image (Tensor): An HDR image tensor of shape (B, C, H, W).
        - predicted_weights (Tensor): A tensor of shape (B, 3) containing dynamic weights for R, G, B channels.
        
        - ldr_image (Tensor): The LDR image resulting from the application of Reinhard tone mapping.
        """
        # Ensure predicted_weights is broadcastable to the shape of hdr_image
        weights = predicted_weights.unsqueeze(0)# (4) -> (1,4)으로 만들기
        luminance = 0.2126 * hdr_image[:, 0, :, :] + \
                    0.7152 * hdr_image[:, 1, :, :] + \
                    0.0722 * hdr_image[:, 2, :, :]
        # Reinhard tone mapping
        ldr_luminance = luminance / (1 + luminance / (weights + 1e-6)) # Luminance Factor
        # Scale factor for maintaining color ratios
        scale_factor = ldr_luminance / (luminance + 1e-6)  # Adding a small value to avoid division by zero(입실론 추가해서 10^-6)
        # Apply scale factor to each channel 각 채널에 scale factor 곱해서 ldr_image 만들기
        ldr_image = hdr_image * scale_factor.unsqueeze(1)  # Unsqueeze to match the dimension of hdr_image

        # Optional: scale to the white point
        if self.white_point != 1.0:
            ldr_image *= self.white_point
        return ldr_image

class DynamicLuminanceWeightNetwork(nn.Module):
    def __init__(self):
        super(DynamicLuminanceWeightNetwork, self).__init__()
        # Upsample 층 추가 (입력 이미지를 224x224로 조정)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # ResNet18 모델 불러오기 (마지막 완전 연결층 제외)
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features  # 마지막 완전 연결층의 입력 특성 수
        self.resnet.fc = nn.Identity()  # 마지막 완전 연결층 제거
        
        # ResNet 부분의 파라미터를 freeze
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # 새로운 완전 연결층 추가
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 2)  # 최종 출력 층
        
        # 새로운 완전 연결층 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        # He 초기화 방법을 사용한 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 이미지를 224x224로 조정
        x = self.upsample(x)
        # 수정된 ResNet18을 통과
        x = self.resnet(x)
        # 새로운 완전 연결층을 통과
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        # 첫 번째 출력에 대해서는 Sigmoid 적용 (범위는 이미 [0, 1])
        x[:, 0] = torch.sigmoid(x[:, 0]) * 0.5 + 0.5
        # 두 번째 출력에 대해서는 Sigmoid 적용 후 [0.5, 2.5] 범위로 조정
        x[:, 1] = torch.sigmoid(x[:, 1]) * 0.5 + 0.5 # 0.5 ~ 1.5
        return x
    
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")

def print_model_weights(model):#epoch 끝날때 마다 weight, bias 바뀌는지 확인
    fc4_weight = model.fc4.weight.data
    fc4_bias = model.fc4.bias.data
    print(f"fc4 weight: {fc4_weight}, bias: {fc4_bias}")

def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix. 정확한 예측 matrix 반환

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


            
@smart_inference_mode()
def run(#train에서 epoch 끝날때마다 시행
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size 32 -> 1로 수정 (default)
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None, #기본이 None이네
    dynamic_luminance_network=None, #0220 추가
    dynamic_luminance_network_weights=None #0220 추가
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py train에서 부르는거?
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        #ckpt = torch.load(weights, map_location=device)  # 체크포인트 로드 0220
        model.half() if half else model.float()
        if dynamic_luminance_network is None:
            dynamic_luminance_network = DynamicLuminanceWeightNetwork().to(device).half() if half else DynamicLuminanceWeightNetwork().to(device).float()
            if dynamic_luminance_network is not None:
                checkpoint = torch.load(dynamic_luminance_network_weights)
                dynamic_luminance_network_state_dict = checkpoint['dynamic_luminance_network']
                dynamic_luminance_network.load_state_dict(dynamic_luminance_network_state_dict)
                print("Loaded dynamic_luminance_network weights")
        print("Model weights start validation:")
        print_model_weights(dynamic_luminance_network)
        dynamic_luminance_network.half() if half else dynamic_luminance_network.float() #초기화
        reinhard_tone_mapping = ReinhardToneMapping().to(device)
        bitEstimator = Model_bpp_estimator().to(device) #bpp estimator 인스턴스 생성
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # bitEstimator.half() if half else bitEstimator.float()
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():#bpp는 freeze 한다.
            param.requires_grad = False
    else:  # called directly (train.py에서 부르지말고 val.py를 실행)
        device = select_device(device, batch_size=batch_size)
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)# YOLOv5 MultiBackend class for python inference on various backends
        ckpt = torch.load(weights[0])#, map_location=device) #0220
        dynamic_luminance_network = DynamicLuminanceWeightNetwork().to(device).half() if half else DynamicLuminanceWeightNetwork().to(device).float() #syh edit dfm 로드
        bitEstimator = Model_bpp_estimator().to(device) #bpp estimator 인스턴스 생성
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # bitEstimator.half() if half else bitEstimator.float()
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():#bpp는 freeze 한다.
            param.requires_grad = False
        reinhard_tone_mapping = ReinhardToneMapping().to(device)
        if 'dynamic_luminance_network' in ckpt: #0220
            dynamic_luminance_network.load_state_dict(ckpt['dynamic_luminance_network']) #0220
            print("Loaded dynamic_luminance_network weights@@@@@@")#0220
        else:#0220
            print("No dynamic_luminance_network weights found in checkpoint@@@@@") #0220
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    dynamic_luminance_network.eval() #syh edit
    cuda = device.type != "cpu" #cpu가 아니면 cuda = True
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # mbpp_loss_val = torch.zeros(1, device=device)  # mean bpp_loss_val
    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device) #초기화
    bpp_loss_val = torch.zeros(1, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    # print("Extended_FC 가중치 (validation 시작):")
    # for name, param in downscaling_network.named_parameters():
    #     if "extended_fc" in name:
    #         print(f"{name}: {param.data}")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar): #시작
        callbacks.run("on_val_batch_start")
        with dt[0]: #데이터 전처리(pre-process)
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32 데이터 타입 조절
            #print('im_size : ',im.size())
            # print("paths : ", paths)#경로
            # #이미지 id 추출
            # img_ids = [path.split('/')[-1].split('.')[0].lstrip('0') for path in paths]
            im /= 255  # 0 - 255 to 0.0 - 1.0
            x = im.size()
            dynamic_luminance_weights = dynamic_luminance_network(im)
            print('\ndynamic_luminance_weights : ',dynamic_luminance_weights)
            resized_imgs = []
            for img, factor in zip(im, dynamic_luminance_weights):
                resized_img = reinhard_tone_mapping(img.unsqueeze(0), factor[0])
                kernel_size = 5 if factor[1] >= 0.8 else 3
                resized_img = gaussian_blur(resized_img, kernel_size=kernel_size, sigma=factor[1])
                resized_imgs.append(resized_img)
            im = torch.cat(resized_imgs, dim=0)#합치기
            print("im size : ",im.size())
            im_jpeg = F.interpolate(im, size=(imgsz, imgsz), mode='bilinear', align_corners=False) # YOLOv5에 640x640 크기로 들어감
            resized_imgs_to_ycbcr = TJ_ycbcr.to_ycbcr(im_jpeg.float(), 1.0, half = False) # 0405코드 추가 // RGB 이미지를 YCbcr로 바꾸기(shape은 동일/ 단 픽셀은 [0,1] 값을 가짐)
            resized_imgs_to_ycbcr = torch.clamp(resized_imgs_to_ycbcr, 0, 1)
            #print("resized_imgs_to_ycbcr size : ",resized_imgs_to_ycbcr.size())
            quality = 80
            quantized_dct_y = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 0:1, :, :], quality, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            quantized_dct_cb = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 1:2, :, :], quality, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            quantized_dct_cr = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 2:3, :, :], quality, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_y = TJ_ijg.decompress_coefficients(quantized_dct_y , quality, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_cb = TJ_ijg.decompress_coefficients(quantized_dct_cb, quality, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_cr = TJ_ijg.decompress_coefficients(quantized_dct_cr, quality, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            quantized_dct = torch.cat([quantized_dct_y, quantized_dct_cb, quantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
            dequantized_dct = torch.cat([dequantized_dct_y, dequantized_dct_cb, dequantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
            dequantized_dct = TJ_ycbcr.to_rgb(dequantized_dct, data_range = 1.0, half= False)
            dequantized_dct = torch.clamp(dequantized_dct, 0, 1)
            dequantized_dct = F.interpolate(dequantized_dct, size=(x[2], x[3]), mode='bilinear', align_corners=False) # YOLOv5에 640x640 크기로 들어감
            #print("quantized_dct size : ",quantized_dct.size())
            blocks = TJ_block.blockify(quantized_dct, 8)
            blocks = rearrange(blocks, 'b c p h w -> b c p (h w)') # (B, C, 80 *80, 8, 8) -> (B, C, 80* 80, 64) // 여기서 p는 블록 갯수 (델타 인코딩을 하기 위해서)
            blocks = delta_encode(blocks) # 델타 인코딩 실시 :각 8x8블록의 첫번째 계수를 이용하여 연속된 블록간의 DC 계수 차이만 저장해서 데이터를 더욱 압축) (B, C, 80 * 80, 64) 데이터 압축
            blocks = rearrange(blocks, 'b c p (h w) -> b c p h w', h = 8, w = 8) #델타 인코딩 끝나면 다시 복원하기 (B,C, 80 * 80, 64) -> (B,C, 80*80, 8, 8)
            blocks = TJ_block.deblockify(blocks, (imgsz, imgsz)) 
            blocks = TJ_dct.zigzag(blocks) # (B,C, 640, 640) -> (B, C, L , 64) # zigzag 실시 안에서 8x8블록단위로 다시 처리하고 zigzag 순서에 따라 벡터화 하고있음.
            blocks = rearrange(blocks, "b c n co -> b (c n) co") # 여기서 n은 벡터화된 DCT 계수의 개수
            blocks = (torch.log(torch.abs(blocks) + 1)) / (torch.log(torch.Tensor([2]).to(device))) #로그 스케일 변환(데이터를 더 잘 처리할 수 있게)
            blocks = rearrange(blocks, "b cn co -> (b cn) co")
            # im = F.interpolate(im, scale_factor = 1 / downscaling_factor, mode='bilinear', recompute_scale_factor = True, align_corners=False)
            # im = F.interpolate(im, size=(640, 640), mode='bilinear', align_corners=False)
            nb, _, height, width = im.shape  # batch size, channels, height, width 
            #원본 : (1,3,imgsz,imgsz) syh

        # Inference
        with dt[1]:
            dequantized_dct = dequantized_dct.to(im.dtype)
            preds, train_out = model(dequantized_dct) if compute_loss else (model(dequantized_dct, augment=augment), None)
            pred_code_len=bitEstimator(blocks) # bpp 추정
            bpp_loss=rearrange(pred_code_len,'(b p1) 1  -> b p1',b=quantized_dct.shape[0])
            bpp_loss=torch.sum(bpp_loss, dim = 1)
            bpp_loss=torch.mean(bpp_loss) 
            bpp_loss=bpp_loss / (quantized_dct.shape[1]*quantized_dct.shape[2]*quantized_dct.shape[3])
            print('\nbpp_loss : ',bpp_loss)
            bpp_loss_val += bpp_loss
            #compute_loss가 True라면 손실 계산을 위해 예측과 함께 훈련 출력도 반환(train_out) -> 학습 중에 모델의 성능을 평가하거나, 손실을 계산하기 위해 사용
            #compute_loss가 False라면, 즉 훈련 중이 아니라 순수하게 모델의 추론 성능을 평가하는 경우(검증, 또는 test 단계) agument 옵션을 사용하여 추론 수행
            #augment는 데이터 증강으로, 검증 또는 테스트 시 모델의 일반화 능력을 더 잘 평가하기 위해 사용될 수 있음
            #preds는 예측 결과로, mAP를 측정하기 위해 꼭 필요 / NMS와 같은 후처리 단계를 거쳐 최종적으로 모델의 성능을 평가 하는데 사용
            #mAP는 모델이 객체를 정확하게 검출하고 분류하는 능력을 종합적으로 평가하는 지표로, 검출 모델의 성능을 평가하는데 널리 사용 

        # Loss loss 계산
        if compute_loss:
            loss += compute_loss(train_out, targets, dynamic_luminance_weights)[1]  # box, obj, cls

        # NMS(non-max suppression) 비 최대 억제 (중복된 검출 제거하고 최종 예측 결정하기 위해 NMS 수행)
        # 여러 겹친 검출 박스 중 가장 신뢰도가 높은 박스를 선택하고 나머지 중복 박스를 제거하는 방식으로 작동
        # 검출 성능을 향상시키고 최종 검출 결과의 정확도를 높이는데 기여
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]: #preds를 전처리함(nms를 통해)
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )
            #conf_thres(신뢰도 임계값) / IoU(Intersection over Union)을 기반으로 중복 검출 제거
            #multi_label = 하나의 객체가 여러 클래스에 속할 수 있음을 나타냄
            #max_det :  최대 검출 객체수 제한
            #agnostic : 클래스에 무관하게 NMS를 적용할 것인지?

        # Metrics 성능 지표 계산(예측 <-> 실제 라벨 간의 성능 지표 계산)
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log 텍스트 파일이나 JSON 파일로 예측 결과 저장하고, 필요한 경우 콘솔이나 로그에 정보 기록 
            #필요 X
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        # 이거는 runs/val/exp#에 저장되는것 같다.
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels 이미지 저장
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred 이미지 저장

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds) #배치 끝났다
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")
    print("average bpp : ", bpp_loss_val)
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds 예측 결과를 시각화하여 저장
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t) #val.py돌릴때 끝날때 나오는 멘트 

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    #downscaling_network.float() # 0225 추가
    dynamic_luminance_network.float()
    if not training: #val.py에서 돌렸을때
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist(), bpp_loss_val.cpu() / len(dataloader)), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)