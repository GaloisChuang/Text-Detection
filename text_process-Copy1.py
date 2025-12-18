import os
import cv2
import numpy as np
import re
import logging
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR

# ==========================================
# 1. 設定區域
# ==========================================
INPUT_ROOT_DIR = r"test_result_default"
OUTPUT_ROOT_DIR = r"text_result"
PADDING = 2  # 裁切時往外擴張的像素

# 設定 Logging 級別
logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==========================================
# 2. 影像處理核心函式
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

def clean_text_edges(img_rgba, threshold=180, kernel_size=1):
    if img_rgba.shape[2] != 4: return img_rgba 
    b, g, r, a = cv2.split(img_rgba)
    _, mask_hard = cv2.threshold(a, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_cleaned = cv2.morphologyEx(mask_hard, cv2.MORPH_OPEN, kernel)
    mask_soft = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)
    return cv2.merge((b, g, r, mask_soft))

def get_largest_numbered_png(folder_path):
    files = list(Path(folder_path).glob("*.png"))
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
    return target_file

# ==========================================
# [新增] 3. 方向修正核心邏輯
# ==========================================
def ensure_text_upright(img_rgba, ocr_engine):
    """
    確保圖片中的文字是水平且正向的。
    1. 如果圖片是直長條 (H > W)，先順時針轉 90 度。
    2. 使用 PaddleOCR 的方向分類器檢查是否上下顛倒 (180度)，若是則修正。
    """
    h, w = img_rgba.shape[:2]
    
    # --- 步驟 A: 幾何旋轉 (處理垂直文字) ---
    # 如果高度顯著大於寬度 (例如 1.2 倍)，判定為垂直文字，先轉成橫的
    if h > 1.2 * w:
        # 這裡預設順時針轉 90 度 (大部分垂直文字是由上往下讀)
        # 如果遇到「由下往上」的文字，這步轉完會變成「倒著的橫字」，會被步驟 B 修正
        img_rgba = cv2.rotate(img_rgba, cv2.ROTATE_90_CLOCKWISE)
    
    # --- 步驟 B: AI 方向偵測 (處理倒立文字) ---
    # 為了讓 PaddleOCR 判斷，需轉為 BGR (丟掉 Alpha)
    img_for_cls = preprocess_for_detection(img_rgba, bg_color='black')
    
    try:
        # 只跑 cls (方向分類)，不跑 det (偵測) 和 rec (辨識)，速度很快
        # 預期回傳格式: [[('180', 0.99)]] 或 [[('0', 0.95)]]
        cls_result = ocr_engine.ocr(img_for_cls, det=False, rec=False, cls=True)
        
        if cls_result and cls_result[0]:
            angle_label, score = cls_result[0][0]
            # 如果模型有信心認為是 180 度顛倒 (Paddle 輸出 '180')
            if angle_label == '180' and score > 0.7:
                img_rgba = cv2.rotate(img_rgba, cv2.ROTATE_180)
                
    except Exception as e:
        print(f"Warning: 方向偵測失敗，保持原狀。Err: {e}")

    return img_rgba

# ==========================================
# 4. PaddleOCR 初始化 (已修改)
# ==========================================
print("正在初始化 PaddleOCR 模型...")
ocr_engine = PaddleOCR(
    use_angle_cls=True,  # 【關鍵修改】開啟方向分類器
    lang='ch', 
    show_log=False,
    det_limit_side_len=1280,   
    det_db_thresh=0.2,         
    det_db_box_thresh=0.5,       
    det_db_unclip_ratio=2.2    
)

# ==========================================
# 5. 主處理邏輯
# ==========================================

def main():
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    subdirs = [f for f in Path(INPUT_ROOT_DIR).iterdir() if f.is_dir()]
    print(f"找到 {len(subdirs)} 個資料夾，開始批次處理...")

    for subdir in tqdm(subdirs):
        target_img_path = get_largest_numbered_png(subdir)
        if target_img_path is None: continue

        img_original = imread_unicode(str(target_img_path))
        if img_original is None: continue

        if img_original.shape[2] == 3:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2BGRA)

        h_img, w_img = img_original.shape[:2]
        img_detect = preprocess_for_detection(img_original, bg_color='black')

        # 執行 OCR
        result = ocr_engine.ocr(img_detect, cls=True) # 這裡 cls=True 會幫助辨識準確度，但主要回傳座標

        save_folder = os.path.join(OUTPUT_ROOT_DIR, subdir.name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        img_vis = img_detect.copy()

        if result and result[0]:
            boxes_data = sorted(result[0], key=lambda x: x[0][0][1])

            for i, line in enumerate(boxes_data):
                box = np.array(line[0]).astype(np.float32)
                text_content = line[1][0]
                
                # --- A. 畫圖標記 ---
                pts = box.astype(np.int32)
                cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                text_y = max(20, pts[0][1] - 10)
                cv2.putText(img_vis, f"L{i}", (pts[0][0], text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- B. 裁切與處理 ---
                x_min = int(np.min(box[:, 0]))
                x_max = int(np.max(box[:, 0]))
                y_min = int(np.min(box[:, 1]))
                y_max = int(np.max(box[:, 1]))

                x_min = max(0, x_min - PADDING)
                x_max = min(w_img, x_max + PADDING)
                y_min = max(0, y_min - PADDING)
                y_max = min(h_img, y_max + PADDING)

                if x_max > x_min and y_max > y_min:
                    crop_img = img_original[y_min:y_max, x_min:x_max]

                    # 1. 先做邊緣淨化 (去背)
                    crop_img_cleaned = clean_text_edges(crop_img, threshold=210, kernel_size=3)
                    
                    # 2. 【新增】執行方向修正 (轉正圖片)
                    crop_img_final = ensure_text_upright(crop_img_cleaned, ocr_engine)

                    # 3. 存檔
                    safe_text = re.sub(r'[\\/*?:"<>|]', "", text_content)
                    if not safe_text: safe_text = f"unknown_{i}"
                    
                    layer_filename = f"layer_{i:02d}_{safe_text}.png"
                    imwrite_unicode(os.path.join(save_folder, layer_filename), crop_img_final)

        imwrite_unicode(os.path.join(save_folder, "result_marked.png"), img_vis)

    print("全部處理完成！")

if __name__ == "__main__":
    main()