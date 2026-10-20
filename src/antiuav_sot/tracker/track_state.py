"""
Bang trang thai track (Track state table): mang co kich thuoc co dinh N khe
(slot), moi khe luu trang thai mot track doc lap (muc 3.5, 4.1 bao cao FPGA).

Day la module QUYET DINH pipeline hoat dong o che do single-object (N=1)
hay multi-object (N>1) - khong phai detector hay motion cue. Voi N=1
(Phuong an A), bang nay chi con mot khe duy nhat, vector so sanh IoU
1xM thay vi ma tran NxM, khong con van de quan ly Track ID hay identity
switch (muc 4.2).

Thiet ke huong toi RTL: tat ca truong deu la mang numpy co dtype/shape co
dinh, KHONG dung list/dict Python dong - phan anh dung cach mot bang trong
BRAM se duoc to chuc (mang cac thanh ghi/hang co dia chi co dinh).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.utils.box_ops import INVALID_BOX, make_empty_boxes

# Track ID = -1 nghia la khe dang TRONG (chua co track nao chiem giu).
EMPTY_TRACK_ID = -1


@dataclasses.dataclass
class TrackStateTable:
    """
    boxes:      (n_slots, 4) xyxy - vi tri box gan nhat cua tung track
    track_ids:  (n_slots,) int32 - ma track, EMPTY_TRACK_ID neu khe trong
    ages:       (n_slots,) int32 - so khung hinh LIEN TIEP khong khop
                (dung cho co che het han, muc 3.5)
    active:     (n_slots,) bool - True neu khe dang chua mot track hop le
    confidence: (n_slots,) float32 - confidence cua lan khop gan nhat
                (0.0 neu track duoc du bao boi Kalman ma khong khop that)
    """

    boxes: np.ndarray
    track_ids: np.ndarray
    ages: np.ndarray
    active: np.ndarray
    confidence: np.ndarray
    n_slots: int
    _next_track_id: int = 0

    @classmethod
    def create_empty(cls, n_slots: int) -> "TrackStateTable":
        return cls(
            boxes=make_empty_boxes(n_slots),
            track_ids=np.full((n_slots,), EMPTY_TRACK_ID, dtype=np.int32),
            ages=np.zeros((n_slots,), dtype=np.int32),
            active=np.zeros((n_slots,), dtype=bool),
            confidence=np.zeros((n_slots,), dtype=np.float32),
            n_slots=n_slots,
        )

    def num_active(self) -> int:
        return int(self.active.sum())

    def find_free_slot(self) -> int | None:
        """Tim khe TRONG dau tien de khoi tao track moi (Tang 3: High Score
        Initiation). Tra ve None neu bang da day (het cho, phai bo qua detection
        moi cho den khi co track het han)."""
        free_indices = np.nonzero(~self.active)[0]
        if free_indices.shape[0] == 0:
            return None
        return int(free_indices[0])

    def init_track(self, slot_idx: int, box_xyxy: np.ndarray, confidence: float) -> int:
        """Khoi tao mot track moi tai khe slot_idx, cap Track ID moi.
        Tra ve Track ID vua cap."""
        if self.active[slot_idx]:
            raise ValueError(
                f"Khe {slot_idx} da co track (id={self.track_ids[slot_idx]}), "
                "khong the khoi tao de nghi tai day."
            )
        new_id = self._next_track_id
        self._next_track_id += 1

        self.boxes[slot_idx] = box_xyxy
        self.track_ids[slot_idx] = new_id
        self.ages[slot_idx] = 0
        self.active[slot_idx] = True
        self.confidence[slot_idx] = confidence
        return new_id

    def update_track(self, slot_idx: int, box_xyxy: np.ndarray, confidence: float) -> None:
        """Cap nhat track hien co tai slot_idx voi box vua khop duoc.
        Reset age ve 0 vi vua khop thanh cong khung hinh nay."""
        self.boxes[slot_idx] = box_xyxy
        self.ages[slot_idx] = 0
        self.confidence[slot_idx] = confidence

    def age_track(self, slot_idx: int) -> None:
        """Tang bo dem tuoi track khi khong khop duoc box nao khung hinh nay."""
        self.ages[slot_idx] += 1

    def expire_track(self, slot_idx: int) -> None:
        """Xoa track tai slot_idx, danh dau khe tro lai trang thai trong."""
        self.boxes[slot_idx] = INVALID_BOX
        self.track_ids[slot_idx] = EMPTY_TRACK_ID
        self.ages[slot_idx] = 0
        self.active[slot_idx] = False
        self.confidence[slot_idx] = 0.0

    def check_and_expire_stale_tracks(self, max_track_age: int) -> list[int]:
        """Duyet toan bo N khe, xoa nhung track co age > max_track_age.
        Tra ve danh sach slot_idx vua bi xoa (de log/debug)."""
        expired_slots = []
        for slot_idx in range(self.n_slots):
            if self.active[slot_idx] and self.ages[slot_idx] > max_track_age:
                expired_slots.append(slot_idx)
                self.expire_track(slot_idx)
        return expired_slots

    def get_active_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """Tra ve (boxes, slot_indices) chi cho cac khe active - dung khi tinh
        IoU matrix voi candidate moi (khong can so voi khe trong)."""
        active_indices = np.nonzero(self.active)[0]
        return self.boxes[active_indices], active_indices
