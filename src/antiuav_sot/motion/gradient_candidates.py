"""
gradient_candidates.py
========================
Adapter bien doi output cua motion_candidates_gradient() (gradient global-
motion + IRLS + connected-components tren residual, xem gradient_motion_core.py)
thanh DUNG HOP DONG fixed-size (K,4) xyxy + valid_mask (K,) ma toan bo pipeline
va Simple Tracker FSM dang doi hoi - GIONG HET chu ky goi cua
extract_motion_candidates() cu trong motion/block_threshold.py. Nho vay,
thay the trong pipeline.py chi la doi 1 dong import + 1 loi goi ham, khong
dung gi den fusion/merge.py hay tracker/*.

So voi block_threshold.py (luoi block co dinh, moi block la 1 candidate DOC
LAP, khong bao gio gop cac block ke nhau lai thanh 1 vat), phuong phap nay:
  - Tu bu chuyen dong camera (global affine motion) truoc khi tinh residual,
    nen loc duoc phan lon "motion gia" do camera pan/rung - dieu block-
    threshold KHONG lam duoc (no chi diff hai frame tho).
  - Tra ve bounding box THAT cua tung vung lien thong (connected component)
    thay vi luoi vuong co dinh -> gan voi kich thuoc that cua vat hon.
  - VAN co the phan manh (mot vat cho ra 2-3 blob rieng neu residual dut
    doan o giua) - nen VAN giu buoc gop cum (_cluster_and_merge_boxes) lam
    luoi an toan, khong phu thuoc thuat toan sinh candidate cu the nao.

Loc kich thuoc (min_box_size/max_box_size) va gop cum la hai co che duoc
thao luan truoc do ve xu ly "kich thuoc box cua motion detector khong dam
bao dung" - xem cuoc trao doi ve fusion tier trong lich su hoi thoai.
"""

from __future__ import annotations

import numpy as np

from antiuav_sot.motion.gradient_motion_core import motion_candidates_gradient
from antiuav_sot.utils.box_ops import make_empty_boxes


def _yxyx_to_xyxy(box_yxyx: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    y0, x0, y1, x1 = box_yxyx
    return (float(x0), float(y0), float(x1), float(y1))


def _boxes_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Khoang cach giua hai box (0 neu cham/overlap nhau), tinh tach truc
    x/y roi lay Euclid - dung de quyet dinh co gop cum khong."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _cluster_and_merge_boxes(
    boxes_xyxy: list[tuple[float, float, float, float]],
    gap_threshold: float,
) -> list[tuple[float, float, float, float]]:
    """Gop cac box co khoang cach bien < gap_threshold thanh 1 box hop nhat
    (union bounding box) bang union-find don gian. Day la buoc phong ngua
    phan manh, khong rieng gi cho thuat toan gradient - co the tai su dung
    cho bat ky nguon motion candidate nao trong tuong lai."""
    n = len(boxes_xyxy)
    if n <= 1:
        return list(boxes_xyxy)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_gap(boxes_xyxy[i], boxes_xyxy[j]) < gap_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for member_idxs in clusters.values():
        xs0 = [boxes_xyxy[i][0] for i in member_idxs]
        ys0 = [boxes_xyxy[i][1] for i in member_idxs]
        xs1 = [boxes_xyxy[i][2] for i in member_idxs]
        ys1 = [boxes_xyxy[i][3] for i in member_idxs]
        merged.append((min(xs0), min(ys0), max(xs1), max(ys1)))

    return merged


def gradient_motion_candidates(
    frame_t: np.ndarray,
    frame_t_minus_1: np.ndarray,
    max_motion_candidates: int,
    min_box_size: int,
    max_box_size: int,
    merge_gap_ratio: float = 0.5,
    model: str = "affine",
    irls_iters: int = 3,
    huber_delta: float = 10.0,
    pyramid_levels: int = 2,
    thresh_factor: float = 2.5,
    min_area: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh motion candidate bang gradient global-motion compensation, tra ve
    dung hop dong fixed-size (boxes (K,4) xyxy, valid_mask (K,)) nhu
    extract_motion_candidates() cu - thay the truc tiep tai pipeline.py.

    frame_t, frame_t_minus_1: (H, W) uint8 grayscale, cung kich thuoc.
    min_box_size / max_box_size: loc theo TUNG CHIEU (w va h deu phai nam
        trong [min_box_size, max_box_size]) - loai bo blob qua nho (nhieu)
        hoac qua lon (thuong la vung camera bu dong khong tot, khong phai
        UAV thuc su).
    merge_gap_ratio: cac blob co khoang cach bien < merge_gap_ratio *
        min_box_size se duoc gop thanh 1 candidate.
    """
    result = motion_candidates_gradient(
        frame_t_minus_1.astype(np.float32),
        frame_t.astype(np.float32),
        model=model,
        irls_iters=irls_iters,
        huber_delta=huber_delta,
        pyramid_levels=pyramid_levels,
        thresh_factor=thresh_factor,
        min_area=min_area,
        verbose=False,
    )

    raw_boxes_xyxy = [_yxyx_to_xyxy(b) for b in result["boxes"]]

    gap_threshold = merge_gap_ratio * min_box_size
    merged_boxes = _cluster_and_merge_boxes(raw_boxes_xyxy, gap_threshold)

    residual = result["residual"]
    filtered: list[tuple[tuple[float, float, float, float], float]] = []
    for box in merged_boxes:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if not (min_box_size <= w <= max_box_size and min_box_size <= h <= max_box_size):
            continue
        ix0, iy0 = int(max(x0, 0)), int(max(y0, 0))
        ix1, iy1 = int(min(x1, residual.shape[1])), int(min(y1, residual.shape[0]))
        strength = float(residual[iy0:iy1, ix0:ix1].sum()) if ix1 > ix0 and iy1 > iy0 else 0.0
        filtered.append((box, strength))

    # Giu lai K candidate manh nhat (tong residual lon nhat trong vung box) -
    # cung tinh than "top-K theo tong lon nhat" nhu block_threshold.py cu,
    # de dam bao fixed-size output khong phu thuoc so luong blob thuc te.
    filtered.sort(key=lambda item: item[1], reverse=True)
    filtered = filtered[:max_motion_candidates]

    boxes = make_empty_boxes(max_motion_candidates)
    valid_mask = np.zeros((max_motion_candidates,), dtype=bool)
    for i, (box, _strength) in enumerate(filtered):
        boxes[i] = box
        valid_mask[i] = True

    return boxes, valid_mask
