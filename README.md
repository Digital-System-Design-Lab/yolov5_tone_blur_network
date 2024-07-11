# YOLOv5(Tone Mapping + Gaussian Blur)

- **Project Name**: YOLOv5(Tone Mapping + Gaussian Blur) (sub 실험)
- **Project Member**: 서영홍, 김예지

## 요약

본 연구는 에지 디바이스에서 서버로 이미지를 그대로 보내지 않고, JPEG과 같은 코덱을 이용하여 이미지를 압축 후 서버로 전송함으로써 전송 데이터의 크기를 줄인다.
이미지 압축 및 전송 이전에 이미지 마다의 적합한 Factor로 Tone Mapping, Gaussian Blur 하여 JPEG 압축 효율성을 증가시키고 데이터 전송 효율(Bitrate)과 Machine Vision Task의 정확도(Accuracy) Optimization 한다.
[더 많은 내용은 여기 클릭](abstract.pdf)

## 전체 architecture

![fig1](https://github.com/Digital-System-Design-Lab/yolov5_tone_blur_network/assets/157951085/4fbbbb31-d300-4b06-911e-e90109ee4cfc)

## Tone Blur Network

![fig2](https://github.com/Digital-System-Design-Lab/yolov5_tone_blur_network/assets/157951085/950cea91-227c-4d34-b31a-2e87b0d0f16c)

## inference result

![fig3](https://github.com/Digital-System-Design-Lab/yolov5_tone_blur_network/assets/157951085/f8ece56c-afb0-46ec-b7a1-86ff963962e0)

## 성능 평가

![fig4](https://github.com/Digital-System-Design-Lab/yolov5_tone_blur_network/assets/157951085/cf9bc820-ae3b-455b-8fe0-b0be295d76e1)

## code example
train example

```bash
CUDA_VISIBLE_DEVICES=0 python train2_0602_qf80_8.py --data VOC.yaml --imgsz 512 --hyp hyp.VOC.yaml --batch-size 8 --epochs 50 --weights VOC_epoch49_mAP_0.62045_imgsz_512_hyp_voc.pt --device 0 --project my_test --name my_name
```

inference example

```bash
CUDA_VISIBLE_DEVICES=0 python val2_0602_qf80_8.py --data VOC.yaml --imgsz 512 --batch-size 8 --weights my_weights.pt --device 0
```
