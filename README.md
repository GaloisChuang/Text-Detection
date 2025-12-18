# Text-Detection & Layer Extraction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-red.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

這是一個基於 **PaddleOCR** 開發的高階文字偵測與自動化圖層提取工具。本專案特別針對複雜排版進行優化，能夠自動將破碎的文字偵測框合併為完整的行與段落，並提取出具備透明度的獨立文字圖層。

## 🌟 核心功能

- **多層級文字合併邏輯**：
  - **水平合併 (Horizontal Merge)**：根據 Y 軸重疊率 (IoU)、水平距離及顏色相似度，將單字合併為完整的句子。
  - **垂直合併 (Vertical Merge)**：偵測段落對齊方式（左對齊、置中、右對齊），將行距合理的文字行歸類為同一個段落。
- **色彩感知 (Color-Aware Processing)**：利用 HSV 空間分析文字區域的中位數顏色，確保不同色彩的文字不會被錯誤合併。
- **高品質圖層提取**：
  - 自動進行邊緣清理 (Edge Cleaning) 與去雜點。
  - 支援自動旋轉校正（0°, 90°, 180°），確保提取出的文字圖層方向正確。
- **結構化數據輸出**：自動生成包含座標、內容、顏色與信心度的 `bbox_info.json`。



## 🛠 環境安裝

請確保您的開發環境已安裝 Python 3.8+，並執行以下指令安裝依賴套件：

```bash
pip install paddleocr paddlepaddle opencv-python numpy tqdm
