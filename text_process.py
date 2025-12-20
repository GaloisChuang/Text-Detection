import os
import cv2
import numpy as np
import re
import logging
import json
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR

# ==========================================
# 0. 參數設定區 (Configuration)
# ==========================================

# --- [路徑設定] ---
INPUT_PATH = r"Selected_images"       # 輸入資料夾或圖片路徑
OUTPUT_ROOT_DIR = r"text_result"      # 輸出結果的根目錄名稱

# --- [PaddleOCR 模型設定] ---
OCR_LANG = 'ch'
OCR_DET_LIMIT_SIDE_LEN = 1280
OCR_DET_DB_THRESH = 0.2
OCR_DET_DB_BOX_THRESH = 0.5
OCR_DET_DB_UNCLIP_RATIO = 2.2

# --- [合併邏輯設定 - 水平 (同一行)] ---
MERGE_H_X_THRESH_RATIO = 2.5
MERGE_H_Y_IOU_THRESH = 0.4
MERGE_H_COLOR_THRESH = 100
MERGE_H_ANGLE_THRESH = 10

# --- [合併邏輯設定 - 垂直 (同一段落)] ---
MERGE_V_Y_THRESH_RATIO = 1.0
MERGE_V_HEIGHT_THRESH = 0.8
MERGE_V_COLOR_THRESH = 50
MERGE_V_ALIGN_THRESH = 20
MERGE_V_ANGLE_THRESH = 10

# --- [圖像處理與裁切設定] ---
PADDING = 2
CLEAN_EDGE_THRESHOLD = 180
CLEAN_EDGE_KERNEL_SIZE = 1
FINAL_CROP_THRESHOLD = 210
FINAL_CROP_KERNEL_SIZE = 3

# --- [Layer 輸出：顏色過濾設定] ---
# 彩色文字才用 HSV 距離；白/黑/灰文字會改用 S/V 規則過濾（避免亮色背景漏進來）
FINAL_LAYER_COLOR_THRESH = 60   # 彩色字：距離門檻（越小越嚴格）
FINAL_LAYER_MIN_ALPHA = 10      # 過濾後太淡 alpha 直接砍掉

# 白字/黑字：規則式過濾的「S 門檻」（越小越嚴格，粉色背景更容易被砍掉）
GRAY_TEXT_S_THRESH = 70

# --- [日誌設定] ---
logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==========================================
# 1. 工具函式
# ==========================================

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)

def imwrite_unicode(path, img):
    cv2.imencode('.png', img)[1].tofile(path)

def preprocess_for_detection(img_rgba, bg_color='black'):
    if img_rgba is None:
        return None
    if len(img_rgba.shape) == 2:
        return cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGR)
    if img_rgba.shape[2] == 3:
        return img_rgba
    b, g, r, a = cv2.split(img_rgba)
    foreground = cv2.merge((b, g, r))
    alpha_mask = a.astype(float) / 255.0
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
    if bg_color == 'black':
        background = np.zeros_like(foreground, dtype=float)
    else:
        background = np.ones_like(foreground, dtype=float) * 255.0
    blended = (foreground.astype(float) * alpha_mask) + (background * (1.0 - alpha_mask))
    return blended.astype(np.uint8)

def clean_text_edges(img_rgba, threshold=CLEAN_EDGE_THRESHOLD, kernel_size=CLEAN_EDGE_KERNEL_SIZE):
    if img_rgba.shape[2] != 4:
        return img_rgba
    b, g, r, a = cv2.split(img_rgba)
    _, mask_hard = cv2.threshold(a, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_cleaned = cv2.morphologyEx(mask_hard, cv2.MORPH_OPEN, kernel)
    mask_soft = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)
    return cv2.merge((b, g, r, mask_soft))

def is_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def color_distance(hsv1, hsv2):
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff) * 2
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    if s1 < 30 and s2 < 30:
        return np.sqrt(v_diff**2)
    return np.sqrt(h_diff**2 + s_diff**2 + (v_diff * 0.8)**2)

def hsv_color_distance_map(hsv_img, ref_hsv):
    """
    向量化 HSV 距離（彩色文字用）
    hsv_img: HxWx3 uint8 (OpenCV HSV: H[0..179], S[0..255], V[0..255])
    ref_hsv: (h,s,v)
    return:  HxW float32 distance map
    """
    h = hsv_img[..., 0].astype(np.float32)
    s = hsv_img[..., 1].astype(np.float32)
    v = hsv_img[..., 2].astype(np.float32)

    h2, s2, v2 = map(float, ref_hsv)

    dh = np.abs(h - h2)
    dh = np.minimum(dh, 180.0 - dh) * 2.0
    ds = np.abs(s - s2)
    dv = np.abs(v - v2)

    # 注意：白/黑/灰文字不走這條（會在 process_final_layer 用規則處理）
    return np.sqrt(dh * dh + ds * ds + (dv * 0.8) * (dv * 0.8))

# ==========================================
# 2. TextBBox 類別
# ==========================================
class TextBBox:
    def __init__(self, raw_points, text, score, image_source=None):
        self.raw_points = np.array(raw_points, dtype=np.float32)
        self.text = text
        self.score = score
        self.color = (0, 0, 0)  # HSV (OpenCV)

        self.x_min = np.min(self.raw_points[:, 0])
        self.x_max = np.max(self.raw_points[:, 0])
        self.y_min = np.min(self.raw_points[:, 1])
        self.y_max = np.max(self.raw_points[:, 1])

        self.w_aabb = self.x_max - self.x_min
        self.h_aabb = self.y_max - self.y_min
        self.cx = (self.x_min + self.x_max) / 2
        self.cy = (self.y_min + self.y_max) / 2

        rect = cv2.minAreaRect(self.raw_points)
        (self.rot_cx, self.rot_cy), (w, h), self.angle = rect

        # 強制修正扁平框角度
        if self.w_aabb > self.h_aabb * 1.2:
            self.angle = 0.0

        self.corners = np.int0(cv2.boxPoints(rect))

        if image_source is not None:
            self._extract_main_color(image_source)

    def _extract_main_color(self, image_source):
        mask_geo = np.zeros(image_source.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_geo, [self.corners], 255)

        x, y, w, h = cv2.boundingRect(self.corners)
        x, y = max(0, x), max(0, y)
        roi = image_source[y:y+h, x:x+w]
        roi_mask_geo = mask_geo[y:y+h, x:x+w]

        if roi.size == 0:
            self.color = (0, 0, 0)
            return

        roi_bgr = roi[:, :, :3] if roi.shape[2] == 4 else roi
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        _, mask_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edge_mean = (np.mean(mask_bin[0, :]) + np.mean(mask_bin[-1, :]) +
                     np.mean(mask_bin[:, 0]) + np.mean(mask_bin[:, -1])) / 4

        text_mask = cv2.bitwise_not(mask_bin) if edge_mean > 127 else mask_bin
        final_text_mask = cv2.bitwise_and(text_mask, roi_mask_geo)

        valid_bgr_pixels = roi_bgr[final_text_mask == 255]
        if valid_bgr_pixels.shape[0] > 0:
            pixel_cnt = valid_bgr_pixels.shape[0]
            valid_bgr_reshaped = valid_bgr_pixels.reshape((pixel_cnt, 1, 3))
            valid_hsv_pixels = cv2.cvtColor(valid_bgr_reshaped, cv2.COLOR_BGR2HSV)
            valid_hsv_pixels = valid_hsv_pixels.reshape((pixel_cnt, 3))
            median_hsv = np.median(valid_hsv_pixels, axis=0)
            self.color = tuple(map(int, median_hsv))
        else:
            self.color = (0, 0, 0)

    @property
    def is_vertical(self):
        return self.h_aabb > (self.w_aabb * 0.9)

    @staticmethod
    def from_group(group_list):
        if not group_list:
            return None
        if len(group_list) == 1:
            return group_list[0]

        all_points = []
        for bbox in group_list:
            all_points.extend(bbox.raw_points)
        all_points = np.array(all_points)

        new_x_min, new_y_min = np.min(all_points, axis=0)
        new_x_max, new_y_max = np.max(all_points, axis=0)
        new_raw_points = [
            [new_x_min, new_y_min], [new_x_max, new_y_min],
            [new_x_max, new_y_max], [new_x_min, new_y_max]
        ]

        vertical_count = sum(1 for b in group_list if b.is_vertical)
        is_vert_group = vertical_count == len(group_list)
        has_chinese = any(is_chinese(b.text) for b in group_list)
        is_bottom_up = is_vert_group and not has_chinese

        x_span = max(b.cx for b in group_list) - min(b.cx for b in group_list)
        y_span = max(b.cy for b in group_list) - min(b.cy for b in group_list)

        if x_span > y_span:
            sorted_bboxes = sorted(group_list, key=lambda b: b.cx)
        else:
            sorted_bboxes = sorted(group_list, key=lambda b: b.cy, reverse=True) if is_bottom_up else \
                            sorted(group_list, key=lambda b: b.cy)

        merged_text = " ".join([b.text for b in sorted_bboxes])
        avg_score = sum(b.score for b in group_list) / len(group_list)

        new_bbox = TextBBox(new_raw_points, merged_text, avg_score, image_source=None)
        new_bbox.color = group_list[0].color
        new_bbox.angle = group_list[0].angle
        return new_bbox

# ==========================================
# 3. 合併邏輯
# ==========================================
def merge_horizontal_boxes(bbox_list, x_thresh_ratio, y_iou_thresh, color_thresh, angle_thresh):
    if not bbox_list:
        return []

    bbox_list_by_y = sorted(bbox_list, key=lambda b: b.y_min)
    rows = []
    current_row = [bbox_list_by_y[0]]

    for i in range(1, len(bbox_list_by_y)):
        cand = bbox_list_by_y[i]
        ref = current_row[-1]

        y_inter_min = max(ref.y_min, cand.y_min)
        y_inter_max = min(ref.y_max, cand.y_max)
        inter_h = max(0, y_inter_max - y_inter_min)
        min_h = min(ref.h_aabb, cand.h_aabb)

        is_same_line = (inter_h / (min_h + 1e-5)) > y_iou_thresh
        if is_same_line:
            current_row.append(cand)
        else:
            rows.append(current_row)
            current_row = [cand]

    if current_row:
        rows.append(current_row)

    final_res = []
    for row in rows:
        row = sorted(row, key=lambda b: b.x_min)
        pending = [row[0]]

        for i in range(1, len(row)):
            cand = row[i]
            ref = pending[-1]

            y_inter_min = max(ref.y_min, cand.y_min)
            y_inter_max = min(ref.y_max, cand.y_max)
            inter_h = max(0, y_inter_max - y_inter_min)
            min_h = min(ref.h_aabb, cand.h_aabb)
            is_same_row = (inter_h / (min_h + 1e-5)) > y_iou_thresh

            gap = cand.x_min - ref.x_max
            current_x_thresh = 6.0 if (ref.is_vertical and cand.is_vertical) else x_thresh_ratio
            is_close_x = gap < (max(ref.w_aabb, cand.w_aabb) * current_x_thresh)

            is_color_ok = color_distance(ref.color, cand.color) < color_thresh
            is_angle_ok = abs(ref.angle - cand.angle) < angle_thresh

            if is_same_row and is_close_x and is_color_ok and is_angle_ok:
                pending.append(cand)
            else:
                final_res.append(TextBBox.from_group(pending))
                pending = [cand]

        if pending:
            final_res.append(TextBBox.from_group(pending))

    return final_res

def group_by_columns(bbox_list):
    if not bbox_list:
        return []
    bbox_list = sorted(bbox_list, key=lambda b: b.x_min)
    columns = []
    current_col = [bbox_list[0]]
    col_x_max = bbox_list[0].x_max

    for i in range(1, len(bbox_list)):
        cand = bbox_list[i]
        buffer = 50
        if cand.x_min < (col_x_max + buffer):
            current_col.append(cand)
            col_x_max = max(col_x_max, cand.x_max)
        else:
            columns.append(current_col)
            current_col = [cand]
            col_x_max = cand.x_max

    if current_col:
        columns.append(current_col)
    return columns

def merge_vertical_boxes(bbox_list, y_thresh_ratio, height_thresh, color_thresh, align_thresh, angle_thresh):
    if not bbox_list:
        return []
    bbox_list = sorted(bbox_list, key=lambda b: b.y_min)
    final_res = []
    pending = [bbox_list[0]]

    for i in range(1, len(bbox_list)):
        cand = bbox_list[i]
        ref = pending[-1]

        is_left_align = abs(ref.x_min - cand.x_min) < align_thresh
        is_right_align = abs(ref.x_max - cand.x_max) < align_thresh
        is_center_align = abs(ref.cx - cand.cx) < align_thresh
        is_aligned = is_left_align or is_right_align or is_center_align

        gap = cand.y_min - ref.y_max
        max_h = min(ref.h_aabb, cand.h_aabb)
        current_y_thresh = y_thresh_ratio + (1.5 if (abs(ref.x_min - cand.x_min) < 5 or abs(ref.x_max - cand.x_max) < 5) else 0.0)
        is_close_y = gap < (max_h * current_y_thresh)

        h_ratio = min(ref.h_aabb, cand.h_aabb) / (max(ref.h_aabb, cand.h_aabb) + 1e-5)
        w_ratio = min(ref.w_aabb, cand.w_aabb) / (max(ref.w_aabb, cand.w_aabb) + 1e-5)
        is_similar = (h_ratio > height_thresh) or (w_ratio > 0.7) or (is_aligned and is_close_y)

        is_color_ok = color_distance(ref.color, cand.color) < color_thresh
        is_angle_ok = abs(ref.angle - cand.angle) < angle_thresh

        if is_close_y and is_aligned and is_similar and is_color_ok and is_angle_ok:
            pending.append(cand)
        else:
            final_res.append(TextBBox.from_group(pending))
            pending = [cand]

    if pending:
        final_res.append(TextBBox.from_group(pending))
    return final_res

# ==========================================
# 4. Layer 輸出：去雜訊的核心
# ==========================================
def get_ocr_score(img, ocr_engine):
    try:
        res = ocr_engine.ocr(img, det=False, cls=False, rec=True)
        if res and res[0]:
            text, score = res[0]
            if len(text) == 0:
                return -1.0
            alnum_ratio = sum(c.isalnum() for c in text) / len(text)
            if alnum_ratio < 0.5:
                score -= 0.5
            if len(text) < 2:
                score -= 0.2
            return score
    except:
        pass
    return -1.0

def process_final_layer(bbox, img_original, ocr_engine):
    """
    目標：把 layer 內「不是文字顏色」的 pixel 設為透明，降低雜訊。

    核心策略：
    1) 先用灰階 Otsu 得到 text_mask（形狀）
    2) 再用顏色過濾：
       - 若文字是白/黑/灰（低飽和）：用 S/V 規則（避免亮色背景 V 很高造成誤保留）
       - 若文字是彩色：用 HSV 距離
    3) alpha = text_mask AND color_mask
    """
    h_img, w_img = img_original.shape[:2]

    x_min = max(0, int(bbox.x_min) - PADDING)
    x_max = min(w_img, int(bbox.x_max) + PADDING)
    y_min = max(0, int(bbox.y_min) - PADDING)
    y_max = min(h_img, int(bbox.y_max) + PADDING)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop_img = img_original[y_min:y_max, x_min:x_max]
    bgr = crop_img[:, :, :3] if crop_img.shape[2] == 4 else crop_img

    # ---- (A) 形狀遮罩：灰階 + Otsu ----
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edge_pixels = np.concatenate([mask_bin[0, :], mask_bin[-1, :], mask_bin[:, 0], mask_bin[:, -1]])
    text_mask = cv2.bitwise_not(mask_bin) if np.mean(edge_pixels) > 127 else mask_bin

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    text_mask = cv2.GaussianBlur(text_mask, (3, 3), 0)

    # ---- (B) 顏色遮罩：依文字類型選方法 ----
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if bbox.color == (0, 0, 0):
        color_mask = np.ones_like(text_mask, dtype=np.uint8) * 255
    else:
        h2, s2, v2 = bbox.color
        S = hsv[..., 1]
        V = hsv[..., 2]

        # Case 1: 低飽和（白/黑/灰字）—用 S/V 規則
        # 這能解你貼的那種「白字 + 亮色背景」：背景 V 高但 S 不低 → 會被砍掉
        if s2 < 40:
            s_thresh = GRAY_TEXT_S_THRESH

            # 白字（亮）
            if v2 > 180:
                v_thresh = max(200, int(v2) - 25)
                color_mask = ((S < s_thresh) & (V > v_thresh)).astype(np.uint8) * 255
            # 黑字（暗）
            else:
                v_thresh = min(90, int(v2) + 25)
                color_mask = ((S < s_thresh) & (V < v_thresh)).astype(np.uint8) * 255

        # Case 2: 彩色字 — 用 HSV 距離
        else:
            dist_map = hsv_color_distance_map(hsv, bbox.color)
            color_mask = (dist_map < FINAL_LAYER_COLOR_THRESH).astype(np.uint8) * 255

    # ---- (C) 合成 alpha，砍掉雜訊 ----
    alpha = cv2.bitwise_and(text_mask, color_mask)
    alpha[alpha < FINAL_LAYER_MIN_ALPHA] = 0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    b, g, r = cv2.split(bgr)
    final_rgba = cv2.merge([b, g, r, alpha])

    # ---- (D) 方向校正（沿用你原本邏輯） ----
    img_for_cls = preprocess_for_detection(final_rgba, bg_color='black')

    h, w = img_for_cls.shape[:2]
    processing_img = img_for_cls
    base_angle = 0
    if h > w * 1.2:
        processing_img = cv2.rotate(img_for_cls, cv2.ROTATE_90_CLOCKWISE)
        base_angle = 90

    try:
        cls_result = ocr_engine.ocr(processing_img, det=False, rec=False, cls=True)
        direction, confidence = ('0', 0.0)
        if cls_result and cls_result[0]:
            direction, confidence = cls_result[0]
    except:
        direction, confidence = ('0', 0.0)

    final_rotation_needed = 0
    if direction == '180' and confidence > 0.85:
        final_rotation_needed = 180
    else:
        score_0 = get_ocr_score(processing_img, ocr_engine)
        score_180 = get_ocr_score(cv2.rotate(processing_img, cv2.ROTATE_180), ocr_engine)
        if score_180 > score_0:
            final_rotation_needed = 180

    final_output = final_rgba
    if base_angle == 90:
        final_output = cv2.rotate(final_output, cv2.ROTATE_90_CLOCKWISE)
    if final_rotation_needed == 180:
        final_output = cv2.rotate(final_output, cv2.ROTATE_180)

    return final_output

# ==========================================
# 5. 輸入與任務管理
# ==========================================
def get_processing_tasks(input_path):
    p = Path(input_path)
    if not p.exists():
        print(f"錯誤：路徑 {input_path} 不存在。")
        return []

    tasks = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

    if p.is_file():
        if p.suffix.lower() in valid_extensions:
            tasks.append((p.stem, p))
        else:
            print(f"跳過：{p.name} 不是支援的圖片格式。")
        return tasks

    if p.is_dir():
        image_files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        if not image_files:
            print(f"錯誤：在 {p.name} 中找不到任何圖片檔案。")
            return []
        print(f"在 {p.name} 中找到 {len(image_files)} 張圖片。")
        for img_path in image_files:
            tasks.append((img_path.stem, img_path))

    tasks.sort(key=lambda x: x[0])
    return tasks

# ==========================================
# 6. 主程式
# ==========================================
print("正在初始化 PaddleOCR 模型...")
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang=OCR_LANG,
    show_log=False,
    det_limit_side_len=OCR_DET_LIMIT_SIDE_LEN,
    det_db_thresh=OCR_DET_DB_THRESH,
    det_db_box_thresh=OCR_DET_DB_BOX_THRESH,
    det_db_unclip_ratio=OCR_DET_DB_UNCLIP_RATIO
)

def main():
    tasks = get_processing_tasks(INPUT_PATH)
    if not tasks:
        print("沒有需要處理的任務。")
        return

    print(f"共發現 {len(tasks)} 個處理任務，開始執行...")

    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    for task_name, img_path in tqdm(tasks):
        img_original = imread_unicode(str(img_path))
        if img_original is None:
            continue

        img_detect = preprocess_for_detection(img_original, bg_color='black')
        result = ocr_engine.ocr(img_detect, cls=True)

        save_folder = os.path.join(OUTPUT_ROOT_DIR, task_name)
        os.makedirs(save_folder, exist_ok=True)

        if result and result[0]:
            raw_data = result[0]

            # Step 1: 轉為 TextBBox
            all_bboxes = []
            for item in raw_data:
                all_bboxes.append(TextBBox(item[0], item[1][0], item[1][1], img_original))

            # 儲存原始 bbox
            raw_bbox_info_list = []
            for i, bbox in enumerate(all_bboxes):
                hsv_px = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
                raw_bbox_info_list.append({
                    "id": i,
                    "text": bbox.text,
                    "score": float(bbox.score),
                    "position": {"x_min": int(bbox.x_min), "y_min": int(bbox.y_min), "x_max": int(bbox.x_max), "y_max": int(bbox.y_max)},
                    "dimension": {"width": int(bbox.w_aabb), "height": int(bbox.h_aabb)},
                    "angle": float(bbox.angle),
                    "center": {"cx": float(bbox.cx), "cy": float(bbox.cy)},
                    "color_hsv": list(bbox.color),
                    "color_bgr": [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]
                })
            with open(os.path.join(save_folder, "bbox_raw.json"), 'w', encoding='utf-8') as f:
                json.dump(raw_bbox_info_list, f, ensure_ascii=False, indent=4)

            # Step 2: Raw Debug 圖
            img_vis_raw = img_detect.copy()
            for bbox in all_bboxes:
                draw_color_hsv = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                draw_color_bgr = cv2.cvtColor(draw_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                cv2.polylines(img_vis_raw, [bbox.corners], True, tuple(map(int, draw_color_bgr)), 2)
            imwrite_unicode(os.path.join(save_folder, "result_raw.png"), img_vis_raw)

            # Step 3: 合併
            merged_h = merge_horizontal_boxes(
                all_bboxes,
                x_thresh_ratio=MERGE_H_X_THRESH_RATIO,
                y_iou_thresh=MERGE_H_Y_IOU_THRESH,
                color_thresh=MERGE_H_COLOR_THRESH,
                angle_thresh=MERGE_H_ANGLE_THRESH
            )
            columns = group_by_columns(merged_h)

            final_bboxes = []
            for col_bboxes in columns:
                final_bboxes.extend(merge_vertical_boxes(
                    col_bboxes,
                    y_thresh_ratio=MERGE_V_Y_THRESH_RATIO,
                    height_thresh=MERGE_V_HEIGHT_THRESH,
                    color_thresh=MERGE_V_COLOR_THRESH,
                    align_thresh=MERGE_V_ALIGN_THRESH,
                    angle_thresh=MERGE_V_ANGLE_THRESH
                ))

            # Step 4: 輸出合併後結果 + layer
            img_vis_merged = img_detect.copy()
            merged_bbox_info_list = []

            for i, bbox in enumerate(final_bboxes):
                cv2.polylines(img_vis_merged, [bbox.corners], True, (0, 255, 0), 2)

                layer_img = process_final_layer(bbox, img_original, ocr_engine)

                safe_text = re.sub(r'[\\/*?:"<>|]', "", bbox.text)
                if not safe_text:
                    safe_text = f"unknown_{i}"
                if len(safe_text) > 30:
                    safe_text = safe_text[:30]

                filename = f"layer_{i:02d}_{safe_text}.png"
                if layer_img is not None:
                    imwrite_unicode(os.path.join(save_folder, filename), layer_img)

                hsv_px = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
                merged_bbox_info_list.append({
                    "id": i,
                    "text": bbox.text,
                    "filename": filename,
                    "position": {"x_min": int(bbox.x_min), "y_min": int(bbox.y_min), "x_max": int(bbox.x_max), "y_max": int(bbox.y_max)},
                    "dimension": {"width": int(bbox.w_aabb), "height": int(bbox.h_aabb)},
                    "angle": float(bbox.angle),
                    "center": {"cx": float(bbox.cx), "cy": float(bbox.cy)},
                    "color_hsv": list(bbox.color),
                    "color_bgr": [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]
                })

            imwrite_unicode(os.path.join(save_folder, "result_marked.png"), img_vis_merged)
            with open(os.path.join(save_folder, "bbox_info.json"), 'w', encoding='utf-8') as f:
                json.dump(merged_bbox_info_list, f, ensure_ascii=False, indent=4)

    print("全部處理完成！")

if __name__ == "__main__":
    main()
