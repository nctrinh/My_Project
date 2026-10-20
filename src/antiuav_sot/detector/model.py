"""
Wrapper cho detector nhe (YOLOv8-n/s) dung Ultralytics, cau hinh rieng cho
anh IR grayscale mot kenh (muc 3.3 bao cao FPGA).

Ultralytics mac dinh ky vong anh 3 kenh RGB (backbone pretrained tren
ImageNet). De giu dung tinh chat "anh dau vao mot kenh, khong phai RGB ba
kenh" cua bai toan Anti-UAV (muc 3.1), lop wrapper nay lap lai kenh
grayscale thanh 3 kenh giong het nhau CHI o buoc dua vao Ultralytics model
(vi Ultralytics chua ho tro sua truc tiep so kenh dau vao mot cach on dinh
qua nhieu phien ban) - kenh thu 4 (INT8 quantization) va cac buoc port
sang FPGA sau nay se lam viec truc tiep tren du lieu 1-kenh that su, KHONG
qua buoc lap kenh nay.

Neu ban muon huan luyen mot backbone CNN nhe tu dau chi voi 1 kenh dau vao
(khong dung pretrained ImageNet), thay lop nay bang mot kien truc tu viet -
gia tri config detector.input_height/width van giu nguyen y nghia.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch.nn as nn


@dataclasses.dataclass
class DetectionResult:
    """Ket qua detector cho MOT anh, da qua NMS, fixed-size.

    boxes:      (max_det_boxes, 4) xyxy trong he toa do anh dau vao detector
                (sau letterbox neu co), INVALID_BOX cho slot rong
    scores:     (max_det_boxes,) float32 confidence
    valid_mask: (max_det_boxes,) bool
    """

    boxes: np.ndarray
    scores: np.ndarray
    valid_mask: np.ndarray


class LightweightYoloDetector:
    """Wrapper mong quanh Ultralytics YOLO, chi lam nhiem vu: nhan anh
    grayscale (H, W) float32 [0,1], tra ve DetectionResult da chuan hoa
    fixed-size. Toan bo phan "training loop" nam trong scripts/train_detector.py,
    KHONG nam trong class nay - class nay chi la interface inference.
    """

    def __init__(
        self,
        model_arch: str = "yolov8n",
        weights_path: str | None = None,
        max_det_boxes: int = 32,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        from ultralytics import YOLO  # import cuc bo de tests khong bat buoc cai ultralytics

        self.max_det_boxes = max_det_boxes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device

        if weights_path is not None:
            self.model = YOLO(weights_path)
        else:
            self.model = YOLO(f"{model_arch}.pt")

        # self.replace_activation(self.model.model, lambda: nn.ReLU(inplace=True))

    def replace_activation(self, module: nn.Module, new_act_fn) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.SiLU):
                setattr(module, name, new_act_fn())
            else:
                self.replace_activation(child, new_act_fn)

    def _grayscale_to_rgb_like(self, image: np.ndarray) -> np.ndarray:
        """Lap kenh grayscale (H, W) thanh (H, W, 3) CHI de tuong thich
        Ultralytics backbone. Xem docstring module ve gioi han cua cach lam nay."""
        if image.ndim == 2:
            image = image[..., None]
        return np.repeat(image, 3, axis=-1)

    def predict(self, image: np.ndarray) -> DetectionResult:
        """
        image: (H, W) float32 [0, 1] hoac uint8 [0, 255], grayscale, DA
               duoc letterbox ve dung (det_input_height, det_input_width)
               truoc khi goi ham nay (xem data/transforms.py).
        """
        rgb_like = self._grayscale_to_rgb_like(image)
        if rgb_like.dtype != np.uint8:
            rgb_like = (np.clip(rgb_like, 0, 1) * 255).astype(np.uint8)

        results = self.model.predict(
            source=rgb_like,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            max_det=self.max_det_boxes,
            device=self.device,
            verbose=False,
        )

        return self._postprocess_ultralytics_result(results[0])

    def _postprocess_ultralytics_result(self, result) -> DetectionResult:
        from antiuav_sot.detector.postprocess import pad_to_fixed_size

        if result.boxes is None or len(result.boxes) == 0:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)

        boxes, scores, valid_mask = pad_to_fixed_size(
            boxes_xyxy, scores, self.max_det_boxes
        )
        return DetectionResult(boxes=boxes, scores=scores, valid_mask=valid_mask)
