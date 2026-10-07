# LINE 靜態貼圖自動化：用 AI 生圖，Python 一鍵去背打包

用 Claude 產提示詞、Gemini 生圖，再用一支 Python 腳本自動完成**切格、去背、疊字、縮放到 LINE 規格、打包成 ZIP**。一套 40 張靜態貼圖，不用一張一張修。

## 檔案

| 檔案 | 說明 |
|---|---|
| `tutorial.pdf` | 完整教學（39 頁）：環境安裝、流程、提示詞寫法、錯誤排除、上架與稅務 |
| `靜態一鍵.py` | 自動處理腳本 |

## 怎麼開始

1. 先讀 `tutorial.pdf` 的第 1～5 章。**也可以把 PDF 直接丟給你的 AI，請它一步步帶你做。**
2. 建一個專案資料夾，把 `靜態一鍵.py` 放進 `系統設定/` 子資料夾（路徑一定要對，教學第 3 章有結構圖）：

```
我的貼圖專案/
├─ 系統設定/靜態一鍵.py
├─ 字型/
├─ input_images/
├─ 文本/
└─ 成品/
```

3. 安裝套件：

```
pip install pillow numpy scipy
```

4. 把 Gemini 生好的 5 張九宮格存進 `input_images/`，命名成 `101-1.png` ～ `101-5.png`。
5. 執行 `靜態一鍵.py`，成品會出現在 `input_images/101_output/` 和 `101_output.zip`。

## 字型

疊字用的字型沒有附在這裡，請自行下載（兩套都是 SIL OFL，可商用），放進 `字型/`：

- 清松手寫體（JasonHandwriting4.ttf）
- 粉圓體（jf-openhuninn-2.1.ttf）

## 注意

- 腳本與教學是 2026 年 8 月的做法，AI 平台改版後可能需要調整，教學第 14 章教你怎麼自己排查。
- 不保證 LINE 審核通過，也不保證任何收入。
- 上架時請據實勾選「使用 AI 生成」。
- 歡迎分享連結，請勿轉售教學或腳本。

## 追蹤更新

Threads：[@yang.biz](https://www.threads.com/@yang.biz)
