"""
Simple Tracker FSM: co che lien ket 4 tang uu tien, dung tinh than bai goc
Tang et al. "Strong Detector with Simple Tracker" (muc 3.5 bao cao FPGA):

    Tang 1 (Middle Score Linking):
        Cap (track, box) dat nguong IoU cao, box tu DETECTOR voi confidence
        o muc trung binh -> khop, cap nhat track (CA vi tri lan kich thuoc,
        detector duoc tin tuong hoan toan).

    Tang 2 (Motion Boxes Linking):
        Voi track/box con chua khop o Tang 1, xet tiep box tu MOTION cue,
        ap dung nguong IoU cao hon de bu cho do tin cay thap hon cua nguon
        nay. Khi khop o tang nay, CHI lay VI TRI (tam) tu box motion, GIU
        NGUYEN kich thuoc cua track cu (_apply_motion_size_preservation) -
        vi motion detector khong dam bao bao phu toan bo vat the, co the
        chi bat duoc mot phan (fragment) cua no. Day chinh la co che giai
        quyet truong hop: vat thu nho lai -> detector conf tut xuong duoi
        tier1_conf_min (khong con duoc xet o Tang 1) NHUNG motion van bat
        duoc vi tri tot -> track van duoc cap nhat dung o Tang nay thay vi
        roi vao Tang 4/mat track.

    Tang 3 (High Score Initiation):
        Box co confidence rat cao (tu detector) nhung chua khop track nao
        -> khoi tao track moi tai khe trong dau tien. Chi tu detector nen
        kich thuoc track moi luon dang tin cay ngay tu dau.

    Tang 4 (Low Score Linking):
        Khop not phan con lai voi nguong noi long hon, buoc du phong cuoi
        cung. Candidate o tang nay co the la detector HOAC motion - ap
        dung CUNG co che giu-nguyen-kich-thuoc o tren khi nguon la motion.

FsmStepResult.slot_matched_source ghi lai NGUON (SOURCE_DETECTOR/
SOURCE_MOTION) cua candidate vua khop moi track trong khung hinh nay (hoac
-1 neu khong khop) - de pipeline.py biet chon measurement_noise phu hop khi
goi ConstantVelocityKalmanBox.update() (quan sat tu motion co the kem tin
cay ve vi tri hon detector).

Day la khoi PHU HOP NHAT voi kien truc FPGA trong toan bo pipeline: toan bo
chi la cac phep so sanh so hoc (IoU) tren du lieu kich thuoc co dinh biet
truoc, khong co logic dieu khien phu thuoc du lieu dong (data-dependent
control flow voi so vong lap khong bien truoc). Phep tinh box-tu-tam-va-
kich-thuoc-cu o Tang 2/4 cung la phep toan hang so - dinh dang IEEE754/
fixed-point don gian, khong them dieu khien dong.

Ghi chu ve cai dat: moi tang duyet TUAN TU qua cac track/candidate con
"chua khop" (unmatched) - day la phan CO tinh dieu khien tuan tu (sequential
logic), khac voi phep tinh IoU matrix o buoc truoc (thuan tuy combinational/
parallel). Trong RTL, phan nay se la mot may trang thai huu han co so buoc
CO DINH (boi vi N va M deu co dinh), khong phai vong lap khong gioi han.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.fusion.merge import SOURCE_DETECTOR, SOURCE_MOTION, CandidateSet
from antiuav_sot.tracker.iou_matrix import compute_masked_iou_matrix
from antiuav_sot.tracker.track_state import TrackStateTable


@dataclasses.dataclass(frozen=True)
class TrackerFsmConfig:
    tier1_iou_threshold: float
    tier1_conf_min: float
    tier2_iou_threshold: float
    tier3_conf_min: float
    tier4_iou_threshold: float
    max_track_age: int


@dataclasses.dataclass
class FsmStepResult:
    """Ket qua cua mot buoc FSM (xu ly mot khung hinh).

    matched_slot_for_candidate: (M,) int32, gia tri = slot_idx neu candidate
        do khop voi mot track, hoac -1 neu khong khop voi track nao.
    tier_for_candidate: (M,) int32, tang nao da khop candidate do (1-4),
        hoac 0 neu khong khop / khong duoc xu ly (bi bo qua vi het slot).
    slot_matched_source: (n_slots,) int32, SOURCE_DETECTOR/SOURCE_MOTION cua
        candidate vua khop track do KHUNG HINH NAY, hoac -1 neu track do
        khong duoc khop khung hinh nay (dang predict-only hoac khe trong).
    newly_initiated_track_ids: danh sach Track ID moi duoc tao o Tang 3.
    expired_track_ids: danh sach Track ID bi xoa do vuot max_track_age.
    """

    matched_slot_for_candidate: np.ndarray
    tier_for_candidate: np.ndarray
    slot_matched_source: np.ndarray
    newly_initiated_track_ids: list[int]
    expired_track_ids: list[int]


TIER_UNMATCHED = 0
TIER_MIDDLE_SCORE = 1
TIER_MOTION_LINK = 2
TIER_HIGH_SCORE_INIT = 3
TIER_LOW_SCORE = 4


def _apply_motion_size_preservation(
    candidate_box: np.ndarray, track_box: np.ndarray
) -> np.ndarray:
    """Khi candidate khop la tu MOTION: lay TAM (center) cua candidate_box
    lam vi tri moi, nhung GIU NGUYEN chieu rong/cao cua track_box hien tai
    (box track truoc khi cap nhat khung hinh nay).

    Ly do: motion detector (frame-diff/gradient-residual-CCL) khong dam bao
    bao phu toan bo vat the - co the chi bat duoc mot phan (fragment), nen
    kich thuoc box cua no khong dang tin cay bang vi tri (vi tri dua tren
    thay doi anh THAT). Gia dinh kich thuoc vat thay doi cham giua cac frame
    lien tiep (hop ly cho bai toan air-to-air) nen giu size cu la lua chon
    an toan hon la tin thang size tu motion.

    QUAN TRONG: ket qua ham nay se duoc dua thang vao
    ConstantVelocityKalmanBox.update() (kalman.py) - filter do KHONG loc
    w/h, chi ghi de _last_wh truc tiep tu box duoc truyen vao, nen buoc
    giu-size-cu nay BAT BUOC phai lam TRUOC khi goi Kalman, khong co co
    che nao khac sua duoc sau do.

    Ca hai box deu xyxy (x0, y0, x1, y1)."""
    cx = (candidate_box[0] + candidate_box[2]) / 2.0
    cy = (candidate_box[1] + candidate_box[3]) / 2.0
    w = track_box[2] - track_box[0]
    h = track_box[3] - track_box[1]
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=candidate_box.dtype,
    )


class SimpleTrackerFSM:
    """Trien khai co che lien ket 4 tang tren mot TrackStateTable co dinh N slot."""

    def __init__(self, config: TrackerFsmConfig, track_state: TrackStateTable):
        self.config = config
        self.track_state = track_state

    def step(self, candidates: CandidateSet) -> FsmStepResult:
        """Xu ly mot khung hinh: nhan CandidateSet (da merge tu detector +
        motion), cap nhat self.track_state tai cho (in-place), tra ve thong
        tin chi tiet ve viec khop de phuc vu debug/visualize/eval.
        """
        n_slots = self.track_state.n_slots
        n_candidates = candidates.boxes.shape[0]

        matched_slot_for_candidate = np.full((n_candidates,), -1, dtype=np.int32)
        tier_for_candidate = np.full((n_candidates,), TIER_UNMATCHED, dtype=np.int32)
        slot_matched_source = np.full((n_slots,), -1, dtype=np.int32)
        slot_matched_this_frame = np.zeros((n_slots,), dtype=bool)

        iou_matrix = compute_masked_iou_matrix(
            self.track_state.boxes,
            self.track_state.active,
            candidates.boxes,
            candidates.valid_mask,
        )  # (n_slots, n_candidates)

        is_detector = candidates.source == SOURCE_DETECTOR
        is_motion = candidates.source == SOURCE_MOTION

        # ---- Tang 1: Middle Score Linking (detector, IoU cao, conf trung binh tro len) ----
        tier1_candidate_mask = (
            candidates.valid_mask
            & is_detector
            & (candidates.scores >= self.config.tier1_conf_min)
        )
        self._match_tier(
            iou_matrix=iou_matrix,
            iou_threshold=self.config.tier1_iou_threshold,
            candidate_eligible_mask=tier1_candidate_mask,
            tier_id=TIER_MIDDLE_SCORE,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Tang 2: Motion Boxes Linking (motion, IoU nguong cao hon) ----
        tier2_candidate_mask = (
            candidates.valid_mask
            & is_motion
            & (tier_for_candidate == TIER_UNMATCHED)
        )
        self._match_tier(
            iou_matrix=iou_matrix,
            iou_threshold=self.config.tier2_iou_threshold,
            candidate_eligible_mask=tier2_candidate_mask,
            tier_id=TIER_MOTION_LINK,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Tang 3: High Score Initiation (detector, conf rat cao, chua khop -> track moi) ----
        newly_initiated_track_ids: list[int] = []
        tier3_candidate_mask = (
            candidates.valid_mask
            & is_detector
            & (candidates.scores >= self.config.tier3_conf_min)
            & (tier_for_candidate == TIER_UNMATCHED)
        )
        for cand_idx in np.nonzero(tier3_candidate_mask)[0]:
            free_slot = self.track_state.find_free_slot()
            if free_slot is None:
                # Bang track da day - bo qua, khong khoi tao duoc them.
                # Tren FPGA day tuong duong tin hieu "table full", khong
                # phai loi, chi la gioi han tai nguyen da cau hinh (N slot).
                break
            new_id = self.track_state.init_track(
                free_slot,
                candidates.boxes[cand_idx],
                float(candidates.scores[cand_idx]),
            )
            matched_slot_for_candidate[cand_idx] = free_slot
            tier_for_candidate[cand_idx] = TIER_HIGH_SCORE_INIT
            slot_matched_source[free_slot] = SOURCE_DETECTOR
            slot_matched_this_frame[free_slot] = True
            newly_initiated_track_ids.append(new_id)

        # ---- Tang 4: Low Score Linking (du phong cuoi, nguong noi long) ----
        tier4_candidate_mask = candidates.valid_mask & (tier_for_candidate == TIER_UNMATCHED)
        self._match_tier(
            iou_matrix=iou_matrix,
            iou_threshold=self.config.tier4_iou_threshold,
            candidate_eligible_mask=tier4_candidate_mask,
            tier_id=TIER_LOW_SCORE,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Cap nhat age cho cac track KHONG duoc khop khung hinh nay ----
        for slot_idx in range(n_slots):
            if self.track_state.active[slot_idx] and not slot_matched_this_frame[slot_idx]:
                self.track_state.age_track(slot_idx)

        # ---- Xoa track vuot qua max_track_age ----
        expired_slots = self.track_state.check_and_expire_stale_tracks(
            self.config.max_track_age
        )
        expired_track_ids = expired_slots  # slot_idx da bi expire, id thuc su
        # da bi ghi de EMPTY_TRACK_ID trong track_state; ta chi can bao hieu
        # SO LUONG va VI TRI slot bi xoa cho ben goi (vd de reset Kalman tuong ung).
        for slot_idx in expired_slots:
            slot_matched_source[slot_idx] = -1

        return FsmStepResult(
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            newly_initiated_track_ids=newly_initiated_track_ids,
            expired_track_ids=expired_track_ids,
        )

    def _match_tier(
        self,
        iou_matrix: np.ndarray,
        iou_threshold: float,
        candidate_eligible_mask: np.ndarray,
        tier_id: int,
        candidates: CandidateSet,
        matched_slot_for_candidate: np.ndarray,
        tier_for_candidate: np.ndarray,
        slot_matched_source: np.ndarray,
        slot_matched_this_frame: np.ndarray,
    ) -> None:
        """
        Thuc hien khop GREEDY trong pham vi mot tang: lap qua cac cap
        (track, candidate) theo thu tu IoU giam dan, chon cap hop le dau
        tien (ca track va candidate deu CHUA duoc khop trong khung hinh nay),
        cho den khi khong con cap nao vuot nguong.

        Khi candidate duoc khop la nguon MOTION (source == SOURCE_MOTION),
        box thuc su commit vao track_state duoc dieu chinh qua
        _apply_motion_size_preservation(): giu nguyen kich thuoc track cu,
        chi cap nhat vi tri theo tam cua candidate. Candidate tu DETECTOR
        van duoc tin tuong hoan toan (ca vi tri lan kich thuoc) nhu cu.

        So vong lap toi da = n_slots * n_candidates (CO DINH, biet truoc tai
        thoi diem cau hinh) - phu hop anh xa thanh mot may trang thai co so
        buoc gioi han ro rang, khong phai vong lap "while co dieu kien dong".
        """
        n_slots, n_candidates = iou_matrix.shape

        # Ap dung mask: track chua khop khung hinh nay VA candidate hop le cho tang nay.
        working_iou = iou_matrix.copy()
        working_iou[~(~slot_matched_this_frame)[:, None].repeat(n_candidates, axis=1)] = 0.0
        # (dong tren giu lai chi cac hang track CHUA matched; slot_matched_this_frame
        #  co the da True do Tang truoc trong CUNG mot lan goi step(), vi du track
        #  vua duoc Tang 1 khop se khong duoc xet lai o Tang 2/4.)
        working_iou = working_iou * candidate_eligible_mask[None, :].astype(np.float32)

        max_iterations = n_slots * n_candidates
        for _ in range(max_iterations):
            if working_iou.size == 0:
                break
            best_idx = np.argmax(working_iou)
            best_slot, best_cand = np.unravel_index(best_idx, working_iou.shape)
            best_value = working_iou[best_slot, best_cand]

            if best_value < iou_threshold:
                break  # khong con cap nao du nguong

            # Khop cap (best_slot, best_cand). Neu nguon la MOTION, dieu
            # chinh box truoc khi commit: giu size track cu, chi lay vi tri
            # tu candidate (xem docstring ham nay va _apply_motion_size_preservation).
            candidate_source = int(candidates.source[best_cand])
            box_to_commit = candidates.boxes[best_cand]
            if candidate_source == SOURCE_MOTION:
                box_to_commit = _apply_motion_size_preservation(
                    box_to_commit, self.track_state.boxes[best_slot]
                )

            self.track_state.update_track(
                int(best_slot),
                box_to_commit,
                float(candidates.scores[best_cand]),
            )
            matched_slot_for_candidate[best_cand] = best_slot
            tier_for_candidate[best_cand] = tier_id
            slot_matched_source[best_slot] = candidate_source
            slot_matched_this_frame[best_slot] = True

            # Loai bo hang/cot vua dung de tranh khop lai
            working_iou[best_slot, :] = 0.0
            working_iou[:, best_cand] = 0.0
