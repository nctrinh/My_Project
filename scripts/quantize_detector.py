#!/usr/bin/env python3
"""
Luong tu hoa (quantization) detector da train, dua checkpoint FP32 ve INT8
truoc khi export sang FPGA toolchain (muc 3.3 bao cao FPGA: "Detector nhe
INT8" la dieu kien tien quyet de dam bao bang thong/tai nguyen DSP kha thi).

Ho tro hai phuong phap:
    PTQ (Post-Training Quantization): nhanh, khong can train lai, chi can
        mot vai batch du lieu de calibrate (uoc luong scale/zero-point).
        Dung khi can ket qua nhanh de kiem tra kha thi.
    QAT (Quantization-Aware Training): fine-tune lai vai epoch VOI cac
        phep gia lap luong tu hoa trong forward pass, thuong giu duoc AP
        cao hon PTQ dang ke, danh doi la ton them thoi gian train.

Ultralytics chua ho tro QAT built-in on-ap dung nghia PyTorch (torch.ao.quantization)
mot cach truc tiep qua API cao cap, nen script nay dung
torch.ao.quantization lam engine chinh, load lai kien truc tu Ultralytics
checkpoint. Day la vung code CAN KIEM TRA KY khi ban chay that su - cac API
quantization cua PyTorch thay doi kha nhieu giua cac phien ban.

Cach dung:
    python scripts/quantize_detector.py --config configs/default.yaml \
        --checkpoint experiments/detector/weights/best.pt --method ptq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from antiuav_sot.utils.config import load_config


def run_ptq(model: torch.nn.Module, calib_loader, calib_batches: int) -> torch.nn.Module:
    """Post-Training Quantization: calibrate scale/zero-point tu vai batch
    du lieu that, khong can gradient/backward."""
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    torch.ao.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= calib_batches:
                break
            model(images)

    torch.ao.quantization.convert(model, inplace=True)
    return model


def run_qat(
    model: torch.nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    device: str,
) -> torch.nn.Module:
    """Quantization-Aware Training: fine-tune voi fake-quant nodes chen vao
    forward pass, giup mo hinh 'thich nghi' voi nhieu do do luong tu hoa
    truoc khi convert that su ve INT8."""
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_detection_loss(outputs, targets)  # xem ghi chu duoi
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"[quantize_detector][QAT] Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    model.eval()
    model.to("cpu")
    torch.ao.quantization.convert(model, inplace=True)
    return model


def compute_detection_loss(outputs, targets):
    """PLACEHOLDER: ham loss thuc te phai tai su dung dung loss cua
    Ultralytics (box loss + cls loss + dfl loss) - Ultralytics khong expose
    ham loss nay o dang de goi doc lap ngoai trainer noi bo cua ho.

    Truoc khi chay QAT that, thay ham nay bang cach hoac (a) tai cau truc
    lai vong lap dung ultralytics.engine.trainer.BaseTrainer voi qconfig
    da gan san vao model, hoac (b) dung thu vien ho tro QAT rieng cho
    YOLO (vd Ultralytics + NNI, hoac Brevitas neu huong toi FPGA/HLS thang
    tu PyTorch). Day la ly do method mac dinh trong configs/default.yaml
    la 'qat' nhung khuyen nghi CHAY THU 'ptq' truoc de co baseline nhanh.
    """
    raise NotImplementedError(
        "compute_detection_loss can duoc thay bang loss thuc te cua "
        "Ultralytics truoc khi chay QAT. Dung --method ptq de chay ngay "
        "voi PTQ (khong can ham loss)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["ptq", "qat"], default=None)
    parser.add_argument(
        "--output", type=str, default="experiments/detector/quantized_int8.pt"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.get("quantization", "method", default="ptq")

    print(f"[quantize_detector] Nap checkpoint: {args.checkpoint}")
    print(f"[quantize_detector] Phuong phap: {method}")

    from ultralytics import YOLO

    yolo_wrapper = YOLO(args.checkpoint)
    torch_model = yolo_wrapper.model

    if method == "ptq":
        calib_batches = cfg.get("quantization", "calib_batches", default=50)
        print(
            "[quantize_detector] CANH BAO: can cung cap calib_loader thuc te "
            "(DataLoader tu du lieu val) truoc khi chay ham nay - hien tai "
            "day la khung code, chua noi voi antiuav_sot.data.dataset."
        )
        # calib_loader = build_calibration_loader(cfg)  # TODO: noi voi dataset that
        # quantized_model = run_ptq(torch_model, calib_loader, calib_batches)
        raise NotImplementedError(
            "Noi calib_loader voi antiuav_sot.data.dataset.AntiUAVDataset "
            "truoc khi chay PTQ that su. Xem TODO trong script nay."
        )
    else:
        qat_epochs = cfg.get("quantization", "qat_epochs", default=10)
        qat_lr = cfg.get("quantization", "qat_lr", default=1e-4)
        device = cfg.get("detector_train", "device", default="cuda:0")
        raise NotImplementedError(
            "QAT can compute_detection_loss() duoc trien khai day du va "
            "train_loader that su. Xem docstring cua compute_detection_loss()."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model.state_dict(), output_path)
    print(f"[quantize_detector] Da luu model INT8 tai: {output_path}")


if __name__ == "__main__":
    main()
