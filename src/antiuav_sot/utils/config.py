"""
Nap va hop nhat configs/default.yaml + configs/fpga_constraints.yaml.

Day la diem duy nhat trong repo doc file YAML truc tiep. Moi module khac
(detector, motion, fusion, tracker...) nhan cac gia tri kich thuoc co dinh
thong qua object Config nay, KHONG tu doc YAML rieng - tranh tinh trang
hai noi doc hai gia tri khac nhau cho cung mot rang buoc.

CAP NHAT (tich hop gradient motion candidates + fusion mechanisms):
  - motion.min_box_size / motion.max_box_size / motion.merge_gap_ratio:
    BAT BUOC - dung boi motion/gradient_candidates.py de loc va gop cum
    candidate. Khong co gia tri mac dinh hop ly (phu thuoc kich thuoc that
    cua UAV muc tieu trong du lieu), nen de raw["motion"][...] bat loi
    KeyError ro rang thay vi am tham dung fallback sai.
  - kalman.motion_measurement_noise: TUY CHON, fallback = measurement_noise
    (nghia la mac dinh KHONG phan biet do tin cay theo nguon, giu hanh vi
    cu). Dat lon hon measurement_noise neu muon Kalman tin vi tri tu motion
    it hon vi tri tu detector.
  - khoi tentative_motion_tracks: TUY CHON CA KHOI (enabled mac dinh False)
    - bat che do khoi tao track moi tu motion sau K khung hinh lien tiep,
    danh cho truong hop vat qua nho de detector tung dat tier3_conf_min.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Loi cau hinh khong hop le - vi du xung dot kich thuoc co dinh."""


@dataclasses.dataclass(frozen=True)
class FpgaConstraints:
    # image
    image_height: int
    image_width: int
    image_channels: int
    # detector
    det_input_height: int
    det_input_width: int
    max_det_boxes: int
    det_conf_threshold: float
    det_nms_iou_threshold: float
    # motion
    block_sum_threshold: int
    max_motion_candidates: int
    motion_min_box_size: int
    motion_max_box_size: int
    motion_merge_gap_ratio: float
    # fusion
    max_total_candidates: int
    # tracker
    n_track_slots: int
    tier1_iou_threshold: float
    tier1_conf_min: float
    tier2_iou_threshold: float
    tier3_conf_min: float
    tier4_iou_threshold: float
    max_track_age: int
    # kalman
    kalman_enabled: bool
    kalman_state_dim: int
    kalman_process_noise: float
    kalman_measurement_noise: float
    kalman_motion_measurement_noise: float
    kalman_max_predict_only_frames: int
    # tentative motion tracks (track moi khoi tao tu motion, tuy chon)
    tentative_motion_tracks_enabled: bool
    max_tentative_motion_tracks: int
    tentative_iou_threshold: float
    tentative_max_miss: int
    motion_track_confirm_frames: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FpgaConstraints":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        image = raw["image"]
        detector = raw["detector"]
        motion = raw["motion"]
        fusion = raw["fusion"]
        tracker = raw["tracker"]
        kalman = raw["kalman"]
        tentative = raw.get("tentative_motion_tracks", {})

        constraints = cls(
            image_height=image["height"],
            image_width=image["width"],
            image_channels=image["channels"],
            det_input_height=detector["input_height"],
            det_input_width=detector["input_width"],
            max_det_boxes=detector["max_det_boxes"],
            det_conf_threshold=detector["conf_threshold"],
            det_nms_iou_threshold=detector["nms_iou_threshold"],
            block_sum_threshold=motion["block_sum_threshold"],
            max_motion_candidates=motion["max_motion_candidates"],
            motion_min_box_size=motion["min_box_size"],
            motion_max_box_size=motion["max_box_size"],
            motion_merge_gap_ratio=motion["merge_gap_ratio"],
            max_total_candidates=fusion["max_total_candidates"],
            n_track_slots=tracker["n_track_slots"],
            tier1_iou_threshold=tracker["tier1_iou_threshold"],
            tier1_conf_min=tracker["tier1_conf_min"],
            tier2_iou_threshold=tracker["tier2_iou_threshold"],
            tier3_conf_min=tracker["tier3_conf_min"],
            tier4_iou_threshold=tracker["tier4_iou_threshold"],
            max_track_age=tracker["max_track_age"],
            kalman_enabled=kalman["enabled"],
            kalman_state_dim=kalman["state_dim"],
            kalman_process_noise=kalman["process_noise"],
            kalman_measurement_noise=kalman["measurement_noise"],
            kalman_motion_measurement_noise=kalman.get(
                "motion_measurement_noise", kalman["measurement_noise"]
            ),
            kalman_max_predict_only_frames=kalman["max_predict_only_frames"],
            tentative_motion_tracks_enabled=tentative.get("enabled", False),
            max_tentative_motion_tracks=tentative.get("max_slots", 0),
            tentative_iou_threshold=tentative.get("iou_threshold", 0.3),
            tentative_max_miss=tentative.get("max_miss", 3),
            motion_track_confirm_frames=tentative.get("confirm_frames", 4),
        )
        constraints.validate()
        return constraints

    def validate(self) -> None:
        """Kiem tra tinh nhat quan giua cac truong - fail som thay vi de
        loi shape xuat hien mo ho o giua pipeline."""
        expected_total = self.max_det_boxes + self.max_motion_candidates
        if self.max_total_candidates != expected_total:
            raise ConfigError(
                f"fusion.max_total_candidates ({self.max_total_candidates}) phai bang "
                f"detector.max_det_boxes + motion.max_motion_candidates "
                f"({self.max_det_boxes} + {self.max_motion_candidates} = {expected_total})"
            )
        if self.image_channels != 1:
            raise ConfigError(
                "image.channels phai = 1 (grayscale IR). Pipeline nay khong ho tro RGB."
            )
        if self.n_track_slots < 1:
            raise ConfigError("tracker.n_track_slots phai >= 1")
        if self.kalman_state_dim != 4:
            raise ConfigError(
                "kalman.state_dim hien chi ho tro = 4 ([x, y, vx, vy]). "
                "Doi gia tri khac se can sua lai KalmanBoxTracker."
            )
        if self.motion_min_box_size <= 0 or self.motion_max_box_size <= self.motion_min_box_size:
            raise ConfigError(
                f"motion.min_box_size ({self.motion_min_box_size}) phai > 0 va "
                f"nho hon motion.max_box_size ({self.motion_max_box_size})"
            )
        if self.motion_merge_gap_ratio < 0:
            raise ConfigError("motion.merge_gap_ratio phai >= 0")
        if self.kalman_motion_measurement_noise < self.kalman_measurement_noise:
            raise ConfigError(
                f"kalman.motion_measurement_noise ({self.kalman_motion_measurement_noise}) "
                f"phai >= kalman.measurement_noise ({self.kalman_measurement_noise}) - "
                "quan sat tu motion khong duoc tin cay HON quan sat tu detector."
            )
        if self.tentative_motion_tracks_enabled:
            if self.max_tentative_motion_tracks <= 0:
                raise ConfigError(
                    "tentative_motion_tracks.max_slots phai >= 1 khi enabled=true"
                )
            if self.motion_track_confirm_frames <= 0:
                raise ConfigError("tentative_motion_tracks.confirm_frames phai >= 1")
            if self.tentative_max_miss < 0:
                raise ConfigError("tentative_motion_tracks.max_miss phai >= 0")


@dataclasses.dataclass(frozen=True)
class Config:
    """Toan bo cau hinh runtime: hyperparameter train/eval + fpga constraints."""

    raw: dict[str, Any]
    fpga: FpgaConstraints
    config_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        fpga_path = path.parent / Path(raw["fpga_constraints_path"]).name \
            if not Path(raw["fpga_constraints_path"]).is_absolute() \
            else Path(raw["fpga_constraints_path"])
        # Neu fpga_constraints_path duoc ghi tuong doi so voi thu muc goc repo
        # (truong hop thong thuong: "configs/fpga_constraints.yaml"), uu tien
        # duong dan do truoc; fallback ve cung thu muc voi default.yaml.
        candidate = Path(raw["fpga_constraints_path"])
        if candidate.exists():
            fpga_path = candidate
        elif (path.parent / candidate.name).exists():
            fpga_path = path.parent / candidate.name
        else:
            raise ConfigError(
                f"Khong tim thay fpga_constraints_path: {raw['fpga_constraints_path']}"
            )

        fpga = FpgaConstraints.from_yaml(fpga_path)
        return cls(raw=raw, fpga=fpga, config_dir=path.parent)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Truy cap an toan vao raw config: cfg.get('detector_train', 'epochs')."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.from_yaml(path)