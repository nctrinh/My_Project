# antiuav-fpga-sot

Pipeline Single-Object Tracking (SOT) cho bai toan Anti-UAV tren anh IR
(thermal), thiet ke theo huong **Tracking-by-Detection**: detector nhe
(YOLOv8-n/s) + motion cue (gradient global-motion compensation + residual
connected-components) + Simple Tracker FSM nhieu tang (IoU/containment) +
Kalman filter tuy chon.

Train/test tren GPU truoc, nhung **moi module deu duoc thiet ke voi kich
thuoc du lieu co dinh** ngay tu dau, de sau nay port sang FPGA ma khong can
viet lai thuat toan tu dau - chi can dich sang HLS/RTL.

## Boi canh thiet ke

Repo nay hien thuc hoa pipeline duoc de xuat trong bao cao kha thi FPGA:
rescope tu "Strong Detector with Simple Tracker" (Tang et al., 3rd Anti-UAV
Workshop, CVPRW 2023) - bo ensemble detector nang (EVA/DINO) va SOT
(MixFormer/OSTrack), giu lai Simple Tracker FSM lam khoi lien ket vi day la
khoi phu hop nhat voi kien truc FPGA (thuan phep so sanh IoU tren du lieu
kich thuoc bien truoc).

Nhanh Motion ban dau theo thiet ke frame-diff + block-threshold (muc 3.2
bao cao goc) **da duoc thay bang Gradient Global-Motion Compensation +
Residual Connected-Components** (xem muc "Hai bien the pipeline" duoi day) -
tu bu duoc chuyen dong camera (pan/rung) truoc khi tinh residual, dieu
frame-diff tho khong lam duoc.

## Cai dat

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` giả định môi trường CUDA 11.6 (torch/torchvision build
`+cu116`) - đổi lại index URL/build phù hợp với GPU đang dùng nếu khác.
`constraints.txt` và `requirements-vitis-frozen.txt` không dùng để cài môi
trường dev: đây là các bản ghim phiên bản tham chiếu cho môi trường
toolchain FPGA (Vitis-AI) - xem chú thích trong từng file trước khi dùng.

Copy du lieu Anti-UAV vao dung vi tri:

```
data/Anti-UAV/Train_Track1_2/train/<seq_name>/
    000001.jpg
    000002.jpg
    ...
    IR_label.json
```

## Cau truc repo

```
configs/, configs1/
    default.yaml              # hyperparameter train/eval + duong dan du lieu/output
    fpga_constraints.yaml     # NGUON SU THAT DUY NHAT cho moi kich thuoc co dinh
src/antiuav_sot/, src1/antiuav_sot/
    data/          # doc IR_label.json, letterbox resize
    motion/        # gradient global-motion compensation + residual CCL -> motion candidates
    detector/      # wrapper YOLOv8-n/s, NMS fixed-size
    fusion/        # merge detector + motion candidates (kich thuoc co dinh)
    tracker/       # Simple Tracker FSM nhieu tang, tentative tracks, Kalman filter
    pipeline/      # rap toan bo pipeline, chay tung frame (streaming)
    utils/         # box ops (vector hoa), config loader, metrics, viz
scripts/           # entry point CLI cho tung buoc (xem duoi)
```

**Khong co thu muc `tests/`** trong repo hien tai, mac du `pytest` co trong
`requirements.txt` va cac comment trong `configs*/fpga_constraints.yaml`
con tham chieu toi `tests/test_pipeline_fixed_shapes.py` (vd validate
`fusion.max_total_candidates == detector.max_det_boxes + motion.max_motion_candidates`)
- bo test nay chua duoc tao/commit vao repo. Rang buoc tuong ung hien duoc
validate o runtime trong `FpgaConstraints.validate()` (`src*/antiuav_sot/utils/config.py`)
thay vi o test.

### Hai bien the pipeline: `src/` + `configs/` vs `src1/` + `configs1/`

Repo hien dang chua **song song hai bien the** cua cung mot pipeline, khac
nhau o muc do xu ly nhanh Motion/Tracker - script nao dung bien the nao
duoc quyet dinh boi `sys.path.insert(..., "src"|"src1")` o dau file, **khong
phai** mot co che chon runtime:

| | `src/` (baseline) | `src1/` (motion-enhanced) |
|---|---|---|
| Config mac dinh | `configs/default.yaml` | `configs1/default.yaml` |
| Motion candidates | Gradient global-motion + residual CCL, gop cum co ban | Them `AdaptiveBoxSizeEstimator` (nguong size dong theo lich su detector), loc ty le canh, "priority anchoring" (neo candidate trung track/tentative dang theo doi) |
| Simple Tracker FSM | 4 tang chuan (Tier 1/2/3/4) | Them tang **Confidence Override** (detector conf rat cao duoc "chiem lai" track dang troi) + khoa `motion_streak` (han che so khung hinh lien tiep track duoc dan dat chi boi motion) + gate "track moi phai o gan track vua mat" (`reinit_near_lost_*`) |
| Dung boi script | `prepare_data.py`, `train_detector.py`, `quantize_detector.py`, `export_onnx.py`, `eval_detector.py`, `eval_pipeline_stages.py`, `eval_tracker_no_motion.py` | `eval_motion.py`, `visualize_motion.py`, `eval_tracker_motion.py`, `visualize_sequence.py` |

`src1/` la ban cap nhat de sua mot loat "van de fusion detector/motion"
phat sinh tu `src/` (vd track bi motion keo lech ma khong co co che cuong
che detector chiem lai, nguong kich thuoc motion box co dinh khong theo kip
UAV o gan/xa...) - xem comment chi tiet trong
`src1/antiuav_sot/pipeline/sot_pipeline.py`,
`src1/antiuav_sot/tracker/simple_tracker_fsm.py` va
`configs1/fpga_constraints.yaml`. `src/` duoc giu lai nguyen trang de co
mot moc so sanh "truoc/sau" khi danh gia cac thay doi nay.

**Luu y khi sua code:** vi hai thu muc KHONG dung chung mot file, sua logic
chung (vd `box_ops.py`, `config.py`) can ap dung o CA HAI noi, hoac quyet
dinh hop nhat lai thanh mot bo duy nhat truoc khi mo rong them.

## Quy trinh su dung

### 1. Kiem tra du lieu + tao manifest train/val

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Quet toan bo sequence, bao loi format neu co (vd so anh khong khop
`len(exist)`), ghi ra `experiments/data_manifest.json` voi danh sach ten
sequence cho train/val (chia theo **sequence**, khong chia theo frame, de
tranh data leakage giua cac frame lien tiep). Manifest nay dung chung cho
ca hai bien the pipeline.

### 2. Train detector

```bash
python scripts/train_detector.py --config configs/default.yaml
```

Tu dong chuyen doi frame sang YOLO format (chi 1 class: `uav`), luu vao
`experiments/yolo_dataset/` - bao gom CA frame co object (exist=1, co box)
LAN frame khong co object (exist=0, negative sample - Ultralytics hieu anh
khong co file `.txt` label tuong ung la "background"), theo ty le dieu
khien boi `data.negative_ratio` trong config. Anh IR grayscale duoc lap
thanh 3 kenh CHI de tuong thich backbone Ultralytics pretrained (xem ghi
chu trong `detector/model.py`).

Augmentation (scale, flip, translate, mosaic, hsv...) doc tu
`configs/default.yaml` -> `detector_train.augmentation`, khong hard-code
trong script - chinh truc tiep trong file YAML de thu nghiem, khong can
sua code. `scale` mac dinh duoc dat cao (0.9) vi UAV la vat the rat nho,
can model thay object o nhieu kich thuoc pixel khac nhau khi train.

**Luu y ve `imgsz`**: Ultralytics train/val mode chi chap nhan `imgsz` la
so nguyen (ep anh ve hinh vuong), du du lieu Anti-UAV la anh chu nhat
512x640 - day la gioi han cua thu vien, khong phai lua chon thiet ke. He
qua la anh se bi them vien (letterbox padding) theo chieu cao khi train,
lam UAV cang nho hon trong anh. Xem ghi chu chi tiet trong
`scripts/train_detector.py`.

### 3. (Tuy chon) Quantize INT8

```bash
python scripts/quantize_detector.py --config configs/default.yaml \
    --checkpoint experiments/detector/weights/best.pt --method ptq
```

**Trang thai hien tai: KHUNG CODE, chua chay duoc end-to-end.** Phan noi
voi DataLoader/loss thuc te cua Ultralytics con `NotImplementedError` co
chu dich - xem docstring trong file de biet chinh xac can hoan thien gi
truoc khi dung (dong bo API `torch.ao.quantization` voi phien ban PyTorch
ban dang dung, va loss function thuc su cua YOLO).

### 4. Export ONNX

```bash
python scripts/export_onnx.py --checkpoint experiments/detector/weights/best.pt
```

Nhanh FP32 hoat dong day du (dung `model.export()` native cua Ultralytics,
`dynamic=False` de giu kich thuoc dau vao co dinh). Nhanh cho checkpoint da
quantize con la khung code, phu thuoc buoc 3.

### 5. Danh gia nhanh Motion rieng le (chon nguong truoc khi vao full pipeline)

```bash
python scripts/eval_motion.py --config configs1/default.yaml \
    --manifest experiments/data_manifest.json --split val
```

Chay rieng `FrameBuffer + gradient_motion_candidates` (khong detector,
khong tracker). Moi frame co GT, lay IoU LON NHAT trong so cac motion box
hop le voi GT (khong trung binh nhieu box). Dung de do nguong
(`min_box_size`, `max_box_size`, `merge_gap_ratio`, `min_strength`,
`strength_metric`, `frame_lag`, `model` affine/translation...) ma khong bi
anh huong boi detector/tracker - hau het nguong co the quet thu qua co
(khong can sua YAML), vd:

```bash
python scripts/eval_motion.py --manifest experiments/data_manifest.json \
    --split val --min-strength 80 --strength-metric mean
```

Debug truc quan mot sequence (xuat video overlay cac motion box sau khi
hop nhat, kem strength tung box):

```bash
python scripts/visualize_motion.py --config configs1/default.yaml \
    --seq-name 20190925_101846_1_2
```

### 6. Danh gia toan bo pipeline (Precision/Success)

Bien the baseline (khong co fix fusion, `src/`):

```bash
python scripts/eval_tracker_no_motion.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val
```

Bien the motion-enhanced (`src1/`):

```bash
python scripts/eval_tracker_motion.py --config configs1/default.yaml \
    --manifest experiments/data_manifest.json --split val
```

Ca hai chay **toan bo** pipeline (detector + motion + fusion + FSM +
Kalman) tung frame tren moi sequence trong tap val, tinh Precision@20px va
Success (AUC) theo chuan OPE - con so quyet dinh "hieu qua cao hay khong"
chu khong phai suy doan tu kien truc. Them `--save-viz` de xuat video
overlay debug cho tung sequence, `--sequence <ten>` de chay mot sequence
duy nhat.

### 6b. Danh gia RIENG detector (doc lap voi tracker)

```bash
python scripts/eval_detector.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val \
    --checkpoint experiments/detector/weights/best.pt
```

Chay `detector.predict()` truc tiep, KHONG di qua motion/fusion/FSM/Kalman.
Bao cao Recall/Precision/mean IoU theo chuan detection, tach rieng frame co
object va frame khong co object (false alarm tren frame rong la loi khac
han voi bo sot muc tieu). Dung script nay TRUOC khi nghi ngo tracker co
van de - neu detector tu no da yeu, sua tracker se khong giai quyet duoc gi.

### 6c. Chan doan pipeline theo tung giai doan (detector vs FSM vs Kalman)

```bash
# Mot sequence cu the (khi da biet sequence nao co van de qua visualize_sequence.py)
python scripts/eval_pipeline_stages.py --config configs/default.yaml \
    --seq-name <ten_sequence> \
    --detector-checkpoint experiments/detector/weights/best.pt

# Toan bo tap val, tong hop
python scripts/eval_pipeline_stages.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val \
    --detector-checkpoint experiments/detector/weights/best.pt
```

Dung bien the `src/` (baseline). Day la cong cu de tra loi cau hoi
"detector bat trung o frame dau nhung track sau do lai lech, loi nam o
dau?" - chay full pipeline nhung ghi lai IoU voi GT tai TUNG giai doan
(detector tho / sau FSM / sau Kalman) cho moi frame, roi tu dong gan co
(flag) cho ba loai bat thuong:

- **`flag_detector_missed_but_object_visible`**: detector tu no da mat muc
  tieu (khong lien quan FSM/Kalman) - neu thay flag nay nhieu, quay lai
  muc 6b/huong dan tuning detector, khong phai sua tracker.
- **`flag_fsm_lost_good_detection`**: detector VAN dung (IoU cao) nhung FSM
  khong khop duoc vao track - thuong xay ra khi track da bi Kalman keo lech
  qua xa TRUOC DO, den luc detection dung quay lai thi IoU voi track (da
  lech) tut duoi ca nguong tier4. Day la nghi pham chinh cho hien tuong
  "object to nhung track khong bat trung theo thoi gian" (va chinh la ly do
  bien the `src1/` them tang Confidence Override).
- **`flag_kalman_diverged_from_fsm`**: FSM khop dung nhung Kalman lam muot
  (smooth) keo vi tri lech khoi quan sat that.

Cung ket hop voi `predict_only_streak` (so frame lien tiep track dang chi
dua vao du bao Kalman, khong co quan sat that) - streak dai truoc mot frame
bi flag `fsm_lost` la dau hieu manh cho thay Kalman da "troi" track ra xa
truoc khi detection dung quay lai.

Ket qua chi tiet tung frame duoc luu JSON tai
`experiments/stage_diagnosis_<seq_name_hoac_split>.json` de phan tich sau.

### 7. Debug truc quan toan bo pipeline tren mot sequence cu the

```bash
python scripts/visualize_sequence.py --config configs1/default.yaml \
    --seq-name <ten_sequence> \
    --detector-checkpoint experiments/detector/weights/best.pt
```

Dung bien the `src1/` (motion-enhanced). Xuat video overlay GT/detector/
motion/track de xem pipeline hoat dong the nao tren mot truong hop cu the,
thay vi chi nhin con so tong hop.

## Nguyen tac thiet ke xuyen suot (quan trong khi sua/mo rong code)

1. **Kich thuoc co dinh, biet truoc tai cau hinh.** Moi mang numpy trong
   pipeline (candidate boxes, track slots...) co shape CO DINH, khong phu
   thuoc du lieu runtime. Slot rong duoc danh dau bang `INVALID_BOX = -1`
   (xem `utils/box_ops.py`), khong dung `None`/list do dai thay doi.

2. **`configs*/fpga_constraints.yaml` la nguon su that duy nhat.** Doi mot
   con so o day (vd `max_det_boxes`) se duoc validate tu dong khi nap
   config (`utils/config.py::FpgaConstraints.validate()`) - neu ban sua
   mot gia tri ma quen sua gia tri lien quan (vd `fusion.max_total_candidates`),
   chuong trinh se bao loi ngay khi khoi dong thay vi loi mo ho giua pipeline.

3. **Vector hoa, khong vong lap tren tung box/pixel.** Xem `utils/box_ops.py`
   (IoU/containment matrix) - day la cac phep tinh se anh xa sang mang PE
   song song tren FPGA, nen phai giu tinh "moi phan tu doc lap" ngay tu ban
   Python. Rieng `motion/gradient_motion_core.py` dung Numba JIT (`@nb.njit`)
   cho vong lap IRLS/affine vi day la buoc tinh tren host (GPU/CPU) khi mo
   phong, khong anh xa truc tiep sang RTL.

4. **Simple Tracker FSM (`tracker/simple_tracker_fsm.py`) la khoi phuc tap
   nhat.** Doc ky docstring cac tang truoc khi sua - thu tu uu tien (tang
   1 Middle Score -> (bien the `src1/`: Confidence Override) -> tang 2
   Motion Link -> tang 3 High Score Init -> tang 4 Low Score) phan anh dung
   logic bai goc, thay doi thu tu se lam sai ket qua ca ve mat thuat toan
   lan mat tuong thich voi thiet ke FPGA da mo ta trong bao cao.

5. **Nhanh Motion dung Gradient Global-Motion Compensation, khong phai
   frame-diff tho.** `motion/frame_buffer.py` chi la mot ring buffer giu
   `frame_lag` frame gan nhat (luu y: docstring dau file cua module nay con
   mo ta thiet ke frame-diff cu, da lac hau so voi class `FrameBuffer` thuc
   te - doc code, khong doc docstring, khi can hieu hanh vi hien tai).
   `motion/gradient_motion_core.py` uoc luong chuyen dong camera toan anh
   (affine/translation, IRLS + Huber weighting, co pyramid) roi
   `motion/gradient_candidates.py` tinh residual sau bu chuyen dong, chay
   connected-components, gop cum fragment, loc kich thuoc/ty le canh truoc
   khi tra ve candidate box fixed-size.

6. **Kalman (`tracker/kalman.py`) chi theo doi TAM box.** Kich thuoc (w, h)
   duoc giu nguyen tu lan quan sat gan nhat, khong di qua buoc loc nao - viec
   "giu size track cu khi khop voi motion" phai duoc xu ly TRUOC do o
   `simple_tracker_fsm.py`.

## Han che da biet (thanh that voi trang thai hien tai)

- `quantize_detector.py`: khung code, chua chay duoc end-to-end (xem muc 3).
- `export_onnx.py --quantized`: phu thuoc buoc tren, cung la khung code.
- Khong co thu muc `tests/` trong repo (xem muc "Cau truc repo" o tren) -
  rang buoc config duoc validate o runtime, khong co test tu dong de giu
  shape co dinh xuyen suot khi refactor.
- Hai bien the `src/`/`configs/` va `src1/`/`configs1/` trung lap phan lon
  code (data/, detector/, fusion/, mot phan utils/) - mot thay doi logic
  chung can duoc ap dung o CA HAI noi, de co nguy co lech dan theo thoi
  gian neu khong hop nhat lai.
- Detector wrapper lap kenh grayscale thanh 3 kenh de dung pretrained
  Ultralytics - day la thoa hiep cho giai doan train/eval GPU, **khong**
  phan anh dung kien truc 1-kenh se chay tren FPGA. Neu muon huan luyen
  backbone 1-kenh tu dau (khong pretrained), can thay `detector/model.py`
  bang kien truc tu viet.
- Chua co script chuyen doi checkpoint ONNX sang dinh dang toolchain FPGA
  cu the (Vitis-AI/hls4ml...) - ngoai pham vi giai doan train/test tren GPU.
  `requirements-vitis-frozen.txt`/`constraints.txt` chi la bang ghim phien
  ban tham chieu cho buoc nay, chua co quy trinh/script dung chung.
