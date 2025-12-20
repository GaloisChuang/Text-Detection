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
# 0. 參數設定區 (Configuration) - 請在此調整參數
# ==========================================

# --- [路徑設定] ---
INPUT_PATH = r"Selected_images"       # 輸入資料夾或圖片路徑
OUTPUT_ROOT_DIR = r"text_result"          # 輸出結果的根目錄名稱

# --- [PaddleOCR 模型設定] ---
# 影響文字檢測的靈敏度與速度
OCR_LANG = 'ch'                           # 語言模型 ('ch', 'en' 等)
OCR_DET_LIMIT_SIDE_LEN = 1280             # 檢測時圖片長邊限制：數字越大檢測越細但在大圖上越慢
OCR_DET_DB_THRESH = 0.2                   # 二值化閾值：越低越能檢測到模糊文字，但也容易有雜訊
OCR_DET_DB_BOX_THRESH = 0.5               # 文字框信心度閾值：低於此分數的框會被過濾掉
OCR_DET_DB_UNCLIP_RATIO = 2.2             # 文字框擴張比例：數字越大，檢測出的框會比文字本身大越多

# --- [合併邏輯設定 - 水平 (同一行)] ---
# 控制如何將左右相鄰的文字框合併成一句話
MERGE_H_X_THRESH_RATIO = 2.5              # 水平距離容忍度：數值越大，隔得越遠的字也會被視為同一句
MERGE_H_Y_IOU_THRESH = 0.4                # 垂直重疊率 (IOU)：判定兩個字是否在「同一行」的基準
MERGE_H_COLOR_THRESH = 100                # 顏色差異容忍度：數值越大，顏色差異大的字也能合併
MERGE_H_ANGLE_THRESH = 10                 # 角度差異容忍度：兩個文字框旋轉角度差超過此值不合併

# --- [合併邏輯設定 - 垂直 (同一段落)] ---
# 控制如何將上下行的文字合併成一個段落
MERGE_V_Y_THRESH_RATIO = 1.0              # 垂直距離容忍度：數值越大，行距越大的行也會被合併
MERGE_V_HEIGHT_THRESH = 0.8               # 高度相似度：判定字體大小是否接近
MERGE_V_COLOR_THRESH = 50                # 顏色差異容忍度 (垂直合併用)
MERGE_V_ALIGN_THRESH = 20                 # 對齊容忍度：判定是否左對齊、右對齊或置中的像素誤差
MERGE_V_ANGLE_THRESH = 10                 # 角度差異容忍度 (垂直合併用)

# --- [圖像處理與裁切設定] ---
PADDING = 2                               # 最終裁切圖片時，在四周保留的額外像素寬度
CLEAN_EDGE_THRESHOLD = 180                # 邊緣清理閾值：用於去背，數值越大去背越強 (0-255)
CLEAN_EDGE_KERNEL_SIZE = 1                # 邊緣清理腐蝕/膨脹核大小：用於去除雜點
FINAL_CROP_THRESHOLD = 210                # 最終輸出圖層的去背閾值 (通常比檢測時嚴格)
FINAL_CROP_KERNEL_SIZE = 3                # 最終輸出圖層的去雜點強度

# --- [日誌設定] ---
logging.getLogger("ppocr").setLevel(logging.WARNING) # 只顯示警告以上訊息，隱藏繁雜的 OCR 輸出

# ==========================================
# 1. 工具函式
# ==========================================

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)

def imwrite_unicode(path, img):
    cv2.imencode('.png', img)[1].tofile(path)

def preprocess_for_detection(img_rgba, bg_color='black'):
    if img_rgba is None: return None
    if len(img_rgba.shape) == 2:
        return cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGR)
    if img_rgba.shape[2] == 3:
        return img_rgba
    b, g, r, a = cv2.split(img_rgba)
    foreground = cv2.merge((b, g, r))
    alpha_mask = a.astype(float) / 255.0
    alpha_mask = np.stack([alpha_mask]*3, axis=-1)
    if bg_color == 'black':
        background = np.zeros_like(foreground, dtype=float)
    else:
        background = np.ones_like(foreground, dtype=float) * 255.0
    blended = (foreground.astype(float) * alpha_mask) + (background * (1.0 - alpha_mask))
    return blended.astype(np.uint8)

def clean_text_edges(img_rgba, threshold=CLEAN_EDGE_THRESHOLD, kernel_size=CLEAN_EDGE_KERNEL_SIZE):
    """
    使用全域參數作為預設值，但也允許呼叫時覆寫
    """
    if img_rgba.shape[2] != 4: return img_rgba 
    b, g, r, a = cv2.split(img_rgba)
    _, mask_hard = cv2.threshold(a, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_cleaned = cv2.morphologyEx(mask_hard, cv2.MORPH_OPEN, kernel)
    mask_soft = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)
    return cv2.merge((b, g, r, mask_soft))

def get_largest_numbered_png(folder_path):
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(list(Path(folder_path).glob(ext)))
    
    if not files: return None
    
    max_num = -1
    target_file = None
    
    for file_path in files:
        filename = file_path.stem
        numbers = re.findall(r'\d+', filename)
        if numbers:
            current_num = max([int(n) for n in numbers])
            if current_num > max_num:
                max_num = current_num
                target_file = file_path
    
    if target_file is None and files:
        return files[0]
        
    return target_file

def is_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff': return True
    return False

def color_distance(hsv1, hsv2):
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff)
    h_diff = h_diff * 2 
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    if s1 < 30 and s2 < 30:
        return np.sqrt(v_diff**2) 
    return np.sqrt(h_diff**2 + s_diff**2 + (v_diff * 0.8)**2)

# ==========================================
# 2. TextBBox 類別
# ==========================================
class TextBBox:
    def __init__(self, raw_points, text, score, image_source=None):
        self.raw_points = np.array(raw_points, dtype=np.float32)
        self.text = text
        self.score = score
        self.color = (0, 0, 0) # HSV Format

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
        
        # 強制修正扁平框的角度
        if self.w_aabb > self.h_aabb * 1.2:
            self.angle = 0.0

        if w < h: w, h = h, w
        self.corners = np.int0(cv2.boxPoints(rect))
        
        if image_source is not None:
            self._extract_main_color(image_source)

    def _extract_main_color(self, image_source):
        # 1. 取得 Bounding Box 的 ROI
        mask_geo = np.zeros(image_source.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_geo, [self.corners], 255)
        
        x, y, w, h = cv2.boundingRect(self.corners)
        x, y = max(0, x), max(0, y)
        roi = image_source[y:y+h, x:x+w]
        roi_mask_geo = mask_geo[y:y+h, x:x+w]

        if roi.size == 0:
            self.color = (0, 0, 0)
            return

        # 2. 轉為灰階以進行文字/背景分離
        if roi.shape[2] == 4:
            roi_bgr = roi[:, :, :3]
        else:
            roi_bgr = roi
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # 3. 使用 Otsu 自動閾值來區分文字與背景
        # 我們假設文字與背景有對比度，Otsu 會找到最佳分割點
        _, mask_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. 判斷誰是文字，誰是背景
        # 計算邊緣像素的平均值來代表背景
        edge_mean = (np.mean(mask_bin[0, :]) + np.mean(mask_bin[-1, :]) + 
                     np.mean(mask_bin[:, 0]) + np.mean(mask_bin[:, -1])) / 4
        
        if edge_mean > 127:
            # 背景是白色(255)，所以文字是黑色(0)，反轉遮罩讓文字變成 255
            text_mask = cv2.bitwise_not(mask_bin)
        else:
            # 背景是黑色(0)，所以文字是白色(255)
            text_mask = mask_bin

        # 5. 結合幾何遮罩（確保只計算 BBox 內的文字）
        final_text_mask = cv2.bitwise_and(text_mask, roi_mask_geo)

        # 6. 只提取文字部分的像素並計算 HSV 中位數
        valid_bgr_pixels = roi_bgr[final_text_mask == 255]

        if valid_bgr_pixels.shape[0] > 0:
            pixel_cnt = valid_bgr_pixels.shape[0]
            valid_bgr_reshaped = valid_bgr_pixels.reshape((pixel_cnt, 1, 3))
            valid_hsv_pixels = cv2.cvtColor(valid_bgr_reshaped, cv2.COLOR_BGR2HSV)
            valid_hsv_pixels = valid_hsv_pixels.reshape((pixel_cnt, 3))
            
            # 使用中位數可以有效排除雜訊像素
            median_hsv = np.median(valid_hsv_pixels, axis=0)
            self.color = tuple(map(int, median_hsv))
        else:
            # 備案：如果沒抓到文字像素，回歸原始邏輯
            self.color = (0, 0, 0)

    @property
    def is_vertical(self):
        return self.h_aabb > (self.w_aabb * 0.9)

    @staticmethod
    def from_group(group_list):
        if not group_list: return None
        if len(group_list) == 1: return group_list[0]

        all_points = []
        for bbox in group_list: all_points.extend(bbox.raw_points)
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

        sorted_bboxes = []
        x_span = max(b.cx for b in group_list) - min(b.cx for b in group_list)
        y_span = max(b.cy for b in group_list) - min(b.cy for b in group_list)
        
        if x_span > y_span: 
             sorted_bboxes = sorted(group_list, key=lambda b: b.cx)
        else: 
            if is_bottom_up:
                sorted_bboxes = sorted(group_list, key=lambda b: b.cy, reverse=True)
            else:
                sorted_bboxes = sorted(group_list, key=lambda b: b.cy)

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
    if not bbox_list: return []

    # --- Phase 1: 先依照 Y 軸將文字分組 ---
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

    # --- Phase 2: 針對每一行，分別做 X 軸排序與合併 ---
    final_res = []
    
    for row in rows:
        row = sorted(row, key=lambda b: b.x_min)
        
        pending = [row[0]]
        for i in range(1, len(row)):
            cand = row[i]
            ref = pending[-1]
            
            # 1. 垂直重疊確認
            y_inter_min = max(ref.y_min, cand.y_min)
            y_inter_max = min(ref.y_max, cand.y_max)
            inter_h = max(0, y_inter_max - y_inter_min)
            min_h = min(ref.h_aabb, cand.h_aabb)
            is_same_row = (inter_h / (min_h + 1e-5)) > y_iou_thresh

            # 2. 水平距離確認
            gap = cand.x_min - ref.x_max
            current_x_thresh = x_thresh_ratio
            if ref.is_vertical and cand.is_vertical:
                current_x_thresh = 6.0 
            is_close_x = gap < (max(ref.w_aabb, cand.w_aabb) * current_x_thresh)
            
            # 3. 顏色確認
            is_color_ok = color_distance(ref.color, cand.color) < color_thresh

            # 4. 角度確認
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
    if not bbox_list: return []
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
    if not bbox_list: return []
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
        current_y_thresh = y_thresh_ratio
        
        if abs(ref.x_min - cand.x_min) < 5 or abs(ref.x_max - cand.x_max) < 5: 
            current_y_thresh += 1.5

        is_close_y = gap < (max_h * current_y_thresh)
        
        h_ratio = min(ref.h_aabb, cand.h_aabb) / (max(ref.h_aabb, cand.h_aabb) + 1e-5)
        w_ratio = min(ref.w_aabb, cand.w_aabb) / (max(ref.w_aabb, cand.w_aabb) + 1e-5)
        
        is_similar = (h_ratio > height_thresh) or (w_ratio > 0.7) or (is_aligned and is_close_y)
        is_color_ok = color_distance(ref.color, cand.color) < color_thresh

        # 角度確認
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
# 4. 最終輸出的處理邏輯
# ==========================================
def remove_background_adaptive(crop_img):
    """
    針對裁切出的文字區塊進行去背，保留文字顏色並將背景設為透明。
    """
    if crop_img is None or crop_img.size == 0:
        return None

    # 1. 轉為灰階來處理遮罩
    if crop_img.shape[2] == 4:
        # 如果本來有 Alpha 通道，先取 BGR
        bgr = crop_img[:, :, :3]
    else:
        bgr = crop_img
        
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. 二值化 (使用自適應閾值或大津演算法)
    # 這裡假設文字與背景有明顯對比。我們先檢測背景是亮還是暗。
    # 透過計算邊緣像素的平均亮度來判斷背景色
    edge_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    avg_bg_brightness = np.mean(edge_pixels)

    if avg_bg_brightness > 127:
        # 背景較亮 -> 文字較暗 (THRESH_BINARY_INV)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # 背景較暗 -> 文字較亮 (THRESH_BINARY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 優化遮罩 (去除雜訊)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 開運算去雜點
    mask = cv2.GaussianBlur(mask, (3, 3), 0) # 平滑邊緣，讓邊緣不會太銳利

    # 4. 合併為 RGBA
    b, g, r = cv2.split(bgr)
    rgba = cv2.merge([b, g, r, mask])
    
    return rgba
    
def process_final_layer(bbox, img_original, ocr_engine):
    h_img, w_img = img_original.shape[:2]
    
    # 1. 取得裁切範圍
    x_min = max(0, int(bbox.x_min) - PADDING)
    x_max = min(w_img, int(bbox.x_max) + PADDING)
    y_min = max(0, int(bbox.y_min) - PADDING)
    y_max = min(h_img, int(bbox.y_max) + PADDING)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop_img = img_original[y_min:y_max, x_min:x_max]
    
    # 2. 準備轉為 RGBA (透明圖層)
    # 判斷輸入圖是否已有 Alpha 通道
    if crop_img.shape[2] == 4:
        bgr = crop_img[:, :, :3]
    else:
        bgr = crop_img
    
    # 3. 建立文字遮罩 (Text Mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # 使用 Otsu 自動計算閾值來分離文字與背景
    _, mask_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 判斷背景色（利用邊緣像素平均值）來決定是否反轉遮罩
    edge_pixels = np.concatenate([mask_bin[0, :], mask_bin[-1, :], mask_bin[:, 0], mask_bin[:, -1]])
    if np.mean(edge_pixels) > 127:
        # 背景是白色，文字是黑色 -> 反轉遮罩，讓文字變成 255 (不透明)
        text_mask = cv2.bitwise_not(mask_bin)
    else:
        # 背景是黑色，文字是白色 -> 維持遮罩
        text_mask = mask_bin

    # 4. 優化遮罩邊緣 (讓文字看起來更平滑)
    # 稍微膨脹一點點以確保文字邊緣完整，然後模糊化產生羽化效果
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    text_mask = cv2.GaussianBlur(text_mask, (3, 3), 0)

    # 5. 合併成透明圖層 (保留原始文字顏色 + 新的文字遮罩)
    b, g, r = cv2.split(bgr)
    final_rgba = cv2.merge([b, g, r, text_mask])

    # --- 後續的 OCR 方向校正邏輯 (維持原樣) ---
    # 使用黑底背景來餵給 OCR 做方向判斷
    img_for_cls = preprocess_for_detection(final_rgba, bg_color='black')
    
    h, w = img_for_cls.shape[:2]
    processing_img = img_for_cls
    base_angle = 0
    if h > w * 1.2:
        processing_img = cv2.rotate(img_for_cls, cv2.ROTATE_90_CLOCKWISE)
        base_angle = 90

    try:
        cls_result = ocr_engine.ocr(processing_img, det=False, rec=False, cls=True)
        direction = '0'; confidence = 0.0
        if cls_result and cls_result[0]:
            direction, confidence = cls_result[0]
    except:
        direction = '0'; confidence = 0.0

    final_rotation_needed = 0 
    if direction == '180' and confidence > 0.85:
        final_rotation_needed = 180
    else:
        score_0 = get_ocr_score(processing_img, ocr_engine)
        score_180 = get_ocr_score(cv2.rotate(processing_img, cv2.ROTATE_180), ocr_engine)
        if score_180 > score_0: final_rotation_needed = 180

    # 執行最後的旋轉
    final_output = final_rgba
    if base_angle == 90:
        final_output = cv2.rotate(final_output, cv2.ROTATE_90_CLOCKWISE)
    if final_rotation_needed == 180:
        final_output = cv2.rotate(final_output, cv2.ROTATE_180)
        
    return final_output

def get_ocr_score(img, ocr_engine):
    try:
        res = ocr_engine.ocr(img, det=False, cls=False, rec=True)
        if res and res[0]:
            text, score = res[0]
            if len(text) == 0: return -1.0
            alnum_ratio = sum(c.isalnum() for c in text) / len(text)
            if alnum_ratio < 0.5: score -= 0.5
            if len(text) < 2: score -= 0.2
            return score
    except:
        pass
    return -1.0

# ==========================================
# 5. 輸入與任務管理
# ==========================================
def get_processing_tasks(input_path):
    p = Path(input_path)
    if not p.exists():
        print(f"錯誤：路徑 {input_path} 不存在。")
        return []

    tasks = []
    # 定義想要處理的圖片副檔名
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

    if p.is_file():
        if p.suffix.lower() in valid_extensions:
            tasks.append((p.stem, p))
        else:
            print(f"跳過：{p.name} 不是支援的圖片格式。")
        return tasks

    if p.is_dir():
        # 取得資料夾下所有檔案
        file_list = list(p.iterdir())
        # 過濾出圖片檔案
        image_files = [f for f in file_list if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if not image_files:
            print(f"錯誤：在 {p.name} 中找不到任何圖片檔案。")
            return []
        
        print(f"在 {p.name} 中找到 {len(image_files)} 張圖片。")
        for img_path in image_files:
            # 使用檔名作為任務名稱 (task_name)，這會決定輸出的子資料夾名稱
            tasks.append((img_path.stem, img_path))
                
    # 依照檔名排序，確保處理順序有邏輯
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

    if not os.path.exists(OUTPUT_ROOT_DIR): os.makedirs(OUTPUT_ROOT_DIR)

    for task_name, img_path in tqdm(tasks):
        img_original = imread_unicode(str(img_path))
        if img_original is None: continue
        
        img_detect = preprocess_for_detection(img_original, bg_color='black')
        
        result = ocr_engine.ocr(img_detect, cls=True)
        
        save_folder = os.path.join(OUTPUT_ROOT_DIR, task_name)
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        if result and result[0]:
            raw_data = result[0]
            
            # Step 1: 轉為物件
            all_bboxes = []
            for item in raw_data:
                bbox = TextBBox(item[0], item[1][0], item[1][1], img_original)
                all_bboxes.append(bbox)

            # 儲存原始 BBox 資訊
            raw_bbox_info_list = []
            for i, bbox in enumerate(all_bboxes):
                hsv_px = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]

                raw_info = {
                    "id": i,
                    "text": bbox.text,
                    "score": float(bbox.score),
                    "position": {
                        "x_min": int(bbox.x_min),
                        "y_min": int(bbox.y_min),
                        "x_max": int(bbox.x_max),
                        "y_max": int(bbox.y_max)
                    },
                    "dimension": {
                        "width": int(bbox.w_aabb),
                        "height": int(bbox.h_aabb)
                    },
                    "angle": float(bbox.angle),
                    "center": {
                        "cx": float(bbox.cx),
                        "cy": float(bbox.cy)
                    },
                    "color_hsv": list(bbox.color),
                    "color_bgr": [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]
                }
                raw_bbox_info_list.append(raw_info)
            
            with open(os.path.join(save_folder, "bbox_raw.json"), 'w', encoding='utf-8') as f:
                json.dump(raw_bbox_info_list, f, ensure_ascii=False, indent=4)

            # Step 2: 輸出 Raw Debug 圖
            img_vis_raw = img_detect.copy()
            for i, bbox in enumerate(all_bboxes):
                draw_color_hsv = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                draw_color_bgr = cv2.cvtColor(draw_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                draw_color = tuple(map(int, draw_color_bgr))
                cv2.polylines(img_vis_raw, [bbox.corners], isClosed=True, color=draw_color, thickness=2)
            imwrite_unicode(os.path.join(save_folder, "result_raw.png"), img_vis_raw)

            # Step 3: 執行合併 (使用上方定義的參數)
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
                merged_col = merge_vertical_boxes(
                    col_bboxes, 
                    y_thresh_ratio=MERGE_V_Y_THRESH_RATIO, 
                    height_thresh=MERGE_V_HEIGHT_THRESH, 
                    color_thresh=MERGE_V_COLOR_THRESH, 
                    align_thresh=MERGE_V_ALIGN_THRESH, 
                    angle_thresh=MERGE_V_ANGLE_THRESH
                )
                final_bboxes.extend(merged_col)

            # Step 4: 輸出合併後結果
            img_vis_merged = img_detect.copy()
            merged_bbox_info_list = []

            for i, bbox in enumerate(final_bboxes):
                vis_color = (0, 255, 0) 
                cv2.polylines(img_vis_merged, [bbox.corners], isClosed=True, color=vis_color, thickness=2)
                
                layer_img = process_final_layer(bbox, img_original, ocr_engine)
                safe_text = re.sub(r'[\\/*?:"<>|]', "", bbox.text)
                if not safe_text: safe_text = f"unknown_{i}"
                if len(safe_text) > 30: safe_text = safe_text[:30]
                
                filename = f"layer_{i:02d}_{safe_text}.png"
                if layer_img is not None:
                    imwrite_unicode(os.path.join(save_folder, filename), layer_img)

                hsv_px = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
                
                bbox_info = {
                    "id": i,
                    "text": bbox.text,
                    "filename": filename,
                    "position": {
                        "x_min": int(bbox.x_min),
                        "y_min": int(bbox.y_min),
                        "x_max": int(bbox.x_max),
                        "y_max": int(bbox.y_max)
                    },
                    "dimension": {
                        "width": int(bbox.w_aabb),
                        "height": int(bbox.h_aabb)
                    },
                    "angle": float(bbox.angle),
                    "center": {
                        "cx": float(bbox.cx),
                        "cy": float(bbox.cy)
                    },
                    "color_hsv": list(bbox.color),
                    "color_bgr": [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]
                }
                merged_bbox_info_list.append(bbox_info)

            imwrite_unicode(os.path.join(save_folder, "result_marked.png"), img_vis_merged)
            
            with open(os.path.join(save_folder, "bbox_info.json"), 'w', encoding='utf-8') as f:
                json.dump(merged_bbox_info_list, f, ensure_ascii=False, indent=4)

    print("全部處理完成！")

if __name__ == "__main__":
    main()