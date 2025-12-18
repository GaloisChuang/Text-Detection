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
# 1. 設定與工具
# ==========================================
INPUT_PATH = r"test_result_default" 
OUTPUT_ROOT_DIR = r"text_result"
PADDING = 2

logging.getLogger("ppocr").setLevel(logging.WARNING)

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

def clean_text_edges(img_rgba, threshold=180, kernel_size=1):
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
        
        # ==========================================
        # [修改] 強制修正扁平框的角度
        # 邏輯：如果 AABB 的寬度明顯大於高度 (例如寬是高的 1.2 倍以上)
        # 則認定這是水平文字，強制將角度歸零。
        # ==========================================
        if self.w_aabb > self.h_aabb * 1.2:
            self.angle = 0.0
        # ==========================================

        if w < h: w, h = h, w
        self.corners = np.int0(cv2.boxPoints(rect))
        
        if image_source is not None:
            self._extract_main_color(image_source)

    def _extract_main_color(self, image_source):
        mask_geo = np.zeros(image_source.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_geo, [self.corners], 255)
        
        x, y, w, h = cv2.boundingRect(self.corners)
        x = max(0, x); y = max(0, y)
        roi = image_source[y:y+h, x:x+w]
        roi_mask_geo = mask_geo[y:y+h, x:x+w]

        if roi.size == 0:
            self.color = (0, 0, 0)
            return

        if roi.shape[2] == 4:
            alpha_channel = roi[:, :, 3]
            valid_mask = (roi_mask_geo == 255) & (alpha_channel == 255)
            roi_bgr = roi[:, :, :3]
        else:
            valid_mask = (roi_mask_geo == 255)
            roi_bgr = roi
        
        valid_bgr_pixels = roi_bgr[valid_mask]

        if valid_bgr_pixels.shape[0] > 0:
            pixel_cnt = valid_bgr_pixels.shape[0]
            valid_bgr_reshaped = valid_bgr_pixels.reshape((pixel_cnt, 1, 3))
            valid_hsv_pixels = cv2.cvtColor(valid_bgr_reshaped, cv2.COLOR_BGR2HSV)
            valid_hsv_pixels = valid_hsv_pixels.reshape((pixel_cnt, 3))
            median_hsv = np.median(valid_hsv_pixels, axis=0)
            self.color = tuple(map(int, median_hsv))
        else:
            if roi.shape[2] == 4:
                 alpha_channel = roi[:, :, 3]
                 fallback_mask = (roi_mask_geo == 255) & (alpha_channel > 0)
                 valid_bgr_pixels = roi[:, :, :3][fallback_mask]
            else:
                 valid_bgr_pixels = roi[roi_mask_geo == 255]
            
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
        # 合併後角度暫時取第一個，若要更精確可取平均
        new_bbox.angle = group_list[0].angle 
        return new_bbox

# ==========================================
# 3. 合併邏輯
# ==========================================
def merge_horizontal_boxes(bbox_list, x_thresh_ratio=2.5, y_iou_thresh=0.4, color_thresh=150, angle_thresh=10):
    """
    修正版：
    1. 先依照 Y 軸分行
    2. 再依照 X 軸合併
    3. [新增] 檢查角度一致性 (Angle Check)
    """
    if not bbox_list: return []

    # --- Phase 1: 先依照 Y 軸將文字分組 (Row Grouping) ---
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

            # 4. [新增] 角度確認
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

def merge_vertical_boxes(bbox_list, y_thresh_ratio=3.0, height_thresh=0.2, color_thresh=80, align_thresh=20, angle_thresh=10):
    """
    修正版：加入角度檢查 (Angle Check)
    """
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
        max_h = max(ref.h_aabb, cand.h_aabb)
        current_y_thresh = y_thresh_ratio
        
        if abs(ref.x_min - cand.x_min) < 5 or abs(ref.x_max - cand.x_max) < 5: 
            current_y_thresh += 1.5

        is_close_y = gap < (max_h * current_y_thresh)
        
        h_ratio = min(ref.h_aabb, cand.h_aabb) / (max(ref.h_aabb, cand.h_aabb) + 1e-5)
        w_ratio = min(ref.w_aabb, cand.w_aabb) / (max(ref.w_aabb, cand.w_aabb) + 1e-5)
        
        is_similar = (h_ratio > height_thresh) or (w_ratio > 0.7) or (is_aligned and is_close_y)
        is_color_ok = color_distance(ref.color, cand.color) < color_thresh

        # [新增] 角度確認
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
def process_final_layer(bbox, img_original, ocr_engine):
    h_img, w_img = img_original.shape[:2]
    
    x_min = max(0, int(bbox.x_min) - PADDING)
    x_max = min(w_img, int(bbox.x_max) + PADDING)
    y_min = max(0, int(bbox.y_min) - PADDING)
    y_max = min(h_img, int(bbox.y_max) + PADDING)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop_img = img_original[y_min:y_max, x_min:x_max]
    crop_img_cleaned = clean_text_edges(crop_img, threshold=210, kernel_size=3)
    img_bgr = preprocess_for_detection(crop_img_cleaned, bg_color='black')
    h, w = img_bgr.shape[:2]
    
    processing_img = img_bgr
    base_angle = 0
    
    if h > w * 1.2:
        processing_img = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        base_angle = 90
    else:
        processing_img = img_bgr
        base_angle = 0

    try:
        cls_result = ocr_engine.ocr(processing_img, det=False, rec=False, cls=True)
        direction = '0'
        confidence = 0.0
        if cls_result and cls_result[0]:
            direction, confidence = cls_result[0]
    except Exception:
        direction = '0'
        confidence = 0.0

    final_img = crop_img_cleaned
    final_rotation_needed = 0 
    
    if direction == '180' and confidence > 0.85:
        final_rotation_needed = 180
    else:
        cand_0 = processing_img
        cand_180 = cv2.rotate(processing_img, cv2.ROTATE_180)
        score_0 = get_ocr_score(cand_0, ocr_engine)
        score_180 = get_ocr_score(cand_180, ocr_engine)
        
        if score_180 > score_0:
            final_rotation_needed = 180
        else:
            final_rotation_needed = 0

    if base_angle == 90:
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
        
    if final_rotation_needed == 180:
        final_img = cv2.rotate(final_img, cv2.ROTATE_180)
        
    return final_img

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

    if p.is_file():
        tasks.append((p.parent.name, p))
        return tasks

    if p.is_dir():
        subdirs = [f for f in p.iterdir() if f.is_dir()]
        
        if subdirs:
            print(f"檢測到根目錄模式，包含 {len(subdirs)} 個子資料夾。")
            for subdir in subdirs:
                target_img = get_largest_numbered_png(subdir)
                if target_img:
                    tasks.append((subdir.name, target_img))
                else:
                    print(f"跳過 {subdir.name}: 找不到圖片。")
        else:
            print("檢測到單一資料夾模式。")
            target_img = get_largest_numbered_png(p)
            if target_img:
                tasks.append((p.name, target_img))
            else:
                print(f"錯誤：在 {p.name} 中找不到任何 png/jpg 圖片。")
                
    return tasks

# ==========================================
# 6. 主程式
# ==========================================
print("正在初始化 PaddleOCR 模型...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False,
    det_limit_side_len=1280, det_db_thresh=0.2, 
    det_db_box_thresh=0.5, det_db_unclip_ratio=2.2)

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

            # --- [修改] 儲存原始 BBox 資訊 (加入 angle) ---
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
                    "angle": float(bbox.angle), # 新增角度資訊
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
            # -------------------------------

            # Step 2: 輸出 Raw Debug 圖
            img_vis_raw = img_detect.copy()
            for i, bbox in enumerate(all_bboxes):
                draw_color_hsv = np.uint8([[[bbox.color[0], bbox.color[1], bbox.color[2]]]])
                draw_color_bgr = cv2.cvtColor(draw_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                draw_color = tuple(map(int, draw_color_bgr))
                cv2.polylines(img_vis_raw, [bbox.corners], isClosed=True, color=draw_color, thickness=2)
            imwrite_unicode(os.path.join(save_folder, "result_raw.png"), img_vis_raw)

            # Step 3: 執行合併 (加入 angle_thresh=10)
            merged_h = merge_horizontal_boxes(all_bboxes, x_thresh_ratio=2.5, y_iou_thresh=0.4, color_thresh=150, angle_thresh=10)
            columns = group_by_columns(merged_h)
            
            final_bboxes = []
            for col_bboxes in columns:
                merged_col = merge_vertical_boxes(col_bboxes, y_thresh_ratio=3.0, height_thresh=0.2, color_thresh=150, align_thresh=20, angle_thresh=10)
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
                    "angle": float(bbox.angle), # 新增角度資訊
                    "center": {
                        "cx": float(bbox.cx),
                        "cy": float(bbox.cy)
                    },
                    "color_hsv": list(bbox.color),
                    "color_bgr": [int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])]
                }
                merged_bbox_info_list.append(bbox_info)

            imwrite_unicode(os.path.join(save_folder, "result_marked.png"), img_vis_merged)
            
            # 寫入 bbox_info.json (合併後的)
            with open(os.path.join(save_folder, "bbox_info.json"), 'w', encoding='utf-8') as f:
                json.dump(merged_bbox_info_list, f, ensure_ascii=False, indent=4)

    print("全部處理完成！")

if __name__ == "__main__":
    main()