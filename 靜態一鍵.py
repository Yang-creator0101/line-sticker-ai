#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
靜態一鍵  ——  LINE 靜態貼圖　去背 → 切格 → （選配疊字）→ 打包
================================================================================
用法：VSCode 按執行鍵，或
    python3 "靜態一鍵.py"            → 掃 input_images 裡所有 5 張齊全的編號
    python3 "靜態一鍵.py" 415 417     → 只做這幾個編號

素材：input_images/{編號}-1 ~ {編號}-5，副檔名 .png/.jpg/.jpeg/.jfif/.webp/.bmp 都認
      （Gemini 有時候會吐 .jfif，這裡直接吃）

★有字／無字＝看檔案自動判斷
      文本/{編號}_plan.json      → 有字，格式 [["文案","動畫類型"], ...]，只取文案
      文本/{編號}_文案.json      → 有字，格式 ["文案1","文案2", ...] 或 {"01":"文案", ...}
      以上都沒有                  → 不疊字（Gemini 圖上已經自帶文字時走這條）
      環境變數 NO_TEXT=1          → 強制不疊字

輸出：
  ① LINE：input_images/{編號}_output/  40張(370×320) + main(240×240) + tab(96×74) → {編號}_output.zip
  ② Redbubble：redbubble_export/{編號}/  原生解析度、乾淨去背、留白邊界

兩種輸出各自判斷做過了沒（ZIP 在不在／資料夾空不空），不會重做。

需要套件：pillow numpy scipy
================================================================================
"""
import os
import re
import sys
import zipfile
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFile, ImageFont

try:
    from scipy import ndimage
except ImportError:
    print("缺套件 scipy。請在命令列執行：\n    pip install scipy\n")
    input("按 Enter 關閉…")
    sys.exit(1)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================ ★固定路徑設定區★ ============================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # line sticker/
INPUT_DIR   = os.path.join(ROOT, "input_images")
RB_OUT_ROOT = os.path.join(ROOT, "redbubble_export")
TEXT_DIR    = os.path.join(ROOT, "文本")

# 認得的素材副檔名。🔴Gemini 有時候存成 .jfif，不加這個 Python 會看不到檔案
IMG_EXT = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp")

# ---- 疊字樣式（黑字白描邊，畫在上方）----
# ── 字型搜尋（骨架版）──────────────────────────────────
#   優先順序：專案的 字型/ 資料夾 → 專案根目錄 → 舊的 JasonHandwriting-master/
#   把下載的字型檔直接丟進「字型」資料夾就會被找到，不用改程式。
def _find_font(filename):
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _d in (_os.path.join(_root, "字型"),
               _os.path.join(_root, "字型", "JasonHandwriting-master"),
               _root,
               _os.path.join(_root, "JasonHandwriting-master")):
        _p = _os.path.join(_d, filename)
        if _os.path.exists(_p):
            return _p
    return _os.path.join(_root, "字型", filename)   # 找不到時回傳預期位置，錯誤訊息才看得懂
# ──────────────────────────────────────────────────

FONT_PATH   = _find_font("JasonHandwriting4.ttf")
FONT_ALT    = _find_font("jf-openhuninn-2.1.ttf")
FONT_SIZE   = 60
FONT_STROKE = 4
TEXT_TOP    = 8            # 文字上緣距畫布頂端
NO_TEXT     = os.environ.get("NO_TEXT", "0") == "1"

# LINE規格尺寸
STK = (370, 320)
MAIN = (240, 240)
TAB = (96, 74)

# 底色/去背判斷用的容錯參數
# 🔴2026-09-05改：不再寫死RGB常數(REF_GREEN/REF_BLUE/REF_WHITE)去比對色距，改成每張圖
# 「量到什麼顏色就用什麼顏色」，因為背景不保證剛好是那個色號(例如白底可能是248,246,250這種
# 偏灰白，不是純255,255,255)；家族判斷(綠/藍/洋紅/白/灰)只看色相關係，不看跟某個定點的距離，
# 之後也比較容易再加新底色，不用每個地方都補一個新的RGB常數。
NEAR_WHITE = 235   # 判定「夠亮、可能是白/格線」的門檻
NEAR_BLACK = 40    # 判定「夠暗、可能是黑/格線」的門檻
ACHROMATIC_SPREAD = 20   # 三通道最大差小於這個值，視為無彩色(白/灰陣營)
SAFETY_BUFFER = 3
MAX_SIDE_TRIM = 90
HOLE_REPORT_PX = 1500

# Redbubble專用參數
RB_MARGIN_RATIO = 0.08   # 角色四周留白 = 畫布邊長 * 此比例(對應Redbubble官方1/8吋留白建議，取寬鬆值)
RB_STRAY_MIN_PX = 60     # 面積小於這個px數的獨立色塊視為雜訊，直接清除
# ==========================================================================


def log(msg):
    print(msg, flush=True)


# ================================================================
# 第一部分：底色偵測 + 裁切 + 去背 + 去白邊
# (這一整段跟原本 sticker_debg_auto.py 的邏輯完全相同，只是合併到同一支檔案裡)
# ================================================================

# 支援的底色家族。之後要「以防萬一」多開一種底色，只要在 _classify_family 多加一個判斷分支，
# 再依需要在 debg/dehalo 的 _bg_removable_mask 補一條規則，不用回頭改一堆地方。
BG_FAMILIES = ("green", "blue", "magenta", "white", "gray")


def _classify_family(bg):
    """依「量到的實際顏色」判斷屬於哪個底色家族。全部用色相關係(誰比誰亮多少)判斷，
    不是跟某個寫死的RGB定點比距離——同一家族不管深淺、是否純色都能判斷到，
    例如白底實際上可能是偏灰白(248,246,250)也一樣算white，不用剛好等於255,255,255。"""
    r0, g0, b0 = int(bg[0]), int(bg[1]), int(bg[2])
    if g0 - max(r0, b0) > 15:
        return "green"
    if b0 - max(r0, g0) > 15:
        return "blue"
    if min(r0, b0) - g0 > 15 and abs(r0 - b0) < 40:
        return "magenta"
    spread = max(r0, g0, b0) - min(r0, g0, b0)
    if spread < ACHROMATIC_SPREAD:
        val = (r0 + g0 + b0) / 3.0
        if val >= 170:
            return "white"
        if val >= 60:
            return "gray"
    return None


def _dominant_color(sample):
    """樣本裡最常出現的顏色(量化到8的倍數後取眾數)。不預先排除任何顏色範圍——
    以前排除白/黑/灰是假設背景不可能是那些顏色，但白底/灰底就是背景本身，
    排除掉反而永遠偵測不到。邊框/角落樣本本來就絕大多數是背景本色，
    直接取眾數已經夠準，不需要再猜哪些像素才算數。"""
    if sample is None or len(sample) == 0:
        return None
    q = (sample // 8 * 8)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    bucket = uniq[np.argmax(counts)]
    # 用量化桶篩出真正屬於這一桶的原始像素，再取平均——避免直接回傳量化後的桶底值
    # (例如255會被桶到248)，讓量到的顏色盡量貼近實際像素值。
    in_bucket = np.all(q == bucket, axis=1)
    real = sample[in_bucket].astype(float).mean(axis=0)
    return int(round(real[0])), int(round(real[1])), int(round(real[2]))


def _edge_ring_sample(a, thickness):
    h, w, _ = a.shape
    t = min(thickness, h // 4, w // 4)
    if t < 2:
        return None
    top = a[0:t, :, :]
    bottom = a[h - t:h, :, :]
    left = a[:, 0:t, :]
    right = a[:, w - t:w, :]
    return np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                            left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)


def _corner_block_sample(a, inset, patch):
    h, w, _ = a.shape
    if inset + patch > min(h, w):
        return None
    corners = [
        a[inset:inset + patch, inset:inset + patch],
        a[inset:inset + patch, w - inset - patch:w - inset],
        a[h - inset - patch:h - inset, inset:inset + patch],
        a[h - inset - patch:h - inset, w - inset - patch:w - inset],
    ]
    return np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)


def _classify_sample(sample):
    """把一份樣本(N×3像素陣列)分類成(家族, 量到的顏色)，量不到回傳 None。"""
    if sample is None:
        return None
    cand = _dominant_color(sample)
    if cand is None:
        return None
    fam = _classify_family(cand)
    if fam is None:
        return None
    return fam, cand


# ================================================================
# 底色偵測：多方法累積式架構（2026-09-13重構）
# ================================================================
# 設計原則：過去每次踩坑都是「寫一個新方法蓋掉舊方法」，但不同踩坑其實需要
# 不同方法根治，沒有一種方法能通殺所有情況——例如「角落色塊只信最淺」解決了
# 786/787，卻讓799(留白鑲邊比取樣深度還厚)出包；反過來"角落色塊只信最深"又會讓
# 786/787出包。正確做法：每一種曾經解決過問題的方法都完整保留成獨立函式，不互相
# 覆蓋，依序嘗試，只有前面的方法「量不到任何底色」(結構性失敗)才輪到下一個。
#
# 這裡誠實記錄一個已經實測過、行不通的想法，避免以後重蹈覆轍：曾經嘗試「每個方法
# 都算一次，用實際去背後的透明比例、或用其他方法互相投票來自動判斷哪個候選才
# 正確」，但實測發現兩種都不可靠——799這張圖真正正確的答案(grid_anchored量到綠色)
# 反而是4種方法裡的少數派，其餘3種方法(corner_shallow/edge_ring/majority_all)全部
# 一致同意錯誤的白色，「多方法投票多數決」在這裡反而會選到錯的；而「去背後保留比例
# 的變化程度」在正確案例間本身就會因構圖不同大幅擺動(標準差可以從0.003到0.03都算
# 正常)，跟799出包當時的量测值域重疊，沒有一個閾值能同時抓到799的錯誤又不誤殺
# 其他正確案例。也就是說，光憑像素本身的統計特徵無法可靠地「自動證明某個候選是
# 錯的」——這不是偷懶，是實測後的結論。
#
# 所以目前的架構是「結構性依序嘗試」而非「算出來再打分數選最好」：
#   ① _bg_candidate_grid_anchored — 2026-09-13新增，解799。幾何抓格線後只在
#      格子內部取樣，9格獨立多數決。已知這個方法在所有測過的案例(見下方
#      detect_bg_color說明)裡，只要抓得到幾何格線就一定跟人工判斷一致，
#      沒有已知的錯誤案例，所以排在第一位。
#   ② _bg_candidate_corner_shallow — 2026-09-12新增，解786/787。角落色塊
#      由淺到深，第一個量得到家族的就採用。只在①抓不到幾何格線時才會用到。
#   ③ _bg_candidate_edge_ring — 更早期就有的四邊細條取樣，由淺到深。
#   ④ _bg_candidate_majority_all — 最原始的兜底：全部樣本混在一起多數決。
# 任何一個新踩坑，只要①②③都無法解決，就在這個列表最後面新增一個新方法，
# 不去修改前面已經驗證過的方法本體——這樣舊踩坑不會因為修新踩坑而復發。


def _bg_candidate_grid_anchored(img_rgb):
    """方法①（2026-09-13新增，解799：留白鑲邊比取樣深度還厚）。

    先用幾何格線抓出真正的格子邊界(跟auto_crop_cells切格時用的是同一組座標)，
    再分別對9格各自的邊緣取一圈樣本、逐格分類、9格多數決。抓不到格線(非3×3
    排版等特殊情況)回傳 None，交給下一個方法。

    為什麼這個方法能同時解決「留白鑲邊」跟「大頭貼頂到格子邊緣」兩種相反的踩坑：
    取樣範圍本來就鎖定在格線內側，不管整張圖外圍有沒有留白鑲邊、鑲邊多厚，都不
    可能取樣到(799的根因)；而每一格各自獨立取樣、以「格數」多數決(不是像素數
    多數決)，只要大多數格子的邊緣還留著背景(大頭貼畫風的規則本來就要求角色
    四周留一圈底色)，就算某一兩格的角色特別頂到某一邊，也不會被壓過去(786/787
    的根因)。兩種底色家族(白線=green、黑線=其他)都各試一次幾何格線，不用事先
    知道底色才能抓格線。"""
    a = np.array(img_rgb)
    min_side = min(a.shape[0], a.shape[1])
    for mode_guess in ("green", "other"):
        g = detect_grid(img_rgb, mode_guess)
        if not g:
            continue
        xs, ys = g
        margin = max(SAFETY_BUFFER, min_side // 170) + 12
        votes = {}
        for r in range(3):
            y0 = ys[r][1] + 1 + margin
            y1 = ys[r + 1][0] - margin
            for c in range(3):
                x0 = xs[c][1] + 1 + margin
                x1 = xs[c + 1][0] - margin
                if y1 - y0 < 20 or x1 - x0 < 20:
                    continue
                cell = a[y0:y1, x0:x1]
                th = max(8, min(cell.shape[0], cell.shape[1]) // 15)
                r_ = _classify_sample(_edge_ring_sample(cell, th))
                if r_:
                    fam, col = r_
                    votes.setdefault(fam, []).append(col)
        if votes:
            best_family = max(votes, key=lambda f: len(votes[f]))
            bg = tuple(int(round(x)) for x in np.mean(votes[best_family], axis=0))
            return best_family, bg
    return None


def _bg_candidate_corner_shallow(img_rgb):
    """方法②（2026-09-12新增，解786/787：大頭貼系列角色頂到格子邊緣）。

    角落色塊由淺到深，第一個量得到家族的就是答案，不跟更深層比多數——786/787
    的根因是深層取樣插進角色本體，跟最淺層(通常才是真背景)比多數會把正確答案
    蓋掉。只在方法①(幾何格線)完全抓不到格線時才輪到這個方法。"""
    a = np.array(img_rgb)
    min_side = min(a.shape[0], a.shape[1])
    for inset_div, patch_div in [(40, 80), (20, 40), (10, 20)]:
        patch = max(12, min_side // patch_div)
        inset = max(patch, min_side // inset_div)
        r = _classify_sample(_corner_block_sample(a, inset, patch))
        if r:
            return r
    return None


def _bg_candidate_edge_ring(img_rgb):
    """方法③（更早期就有）：四邊細條取樣，由淺到深，第一個量得到家族的就採用。"""
    a = np.array(img_rgb)
    min_side = min(a.shape[0], a.shape[1])
    for t in (max(2, min_side // 200), max(4, min_side // 100), max(8, min_side // 50)):
        r = _classify_sample(_edge_ring_sample(a, t))
        if r:
            return r
    return None


def _bg_candidate_majority_all(img_rgb):
    """方法④（最原始的兜底邏輯）：角落色塊＋四邊細條全部樣本混在一起多數決。"""
    a = np.array(img_rgb)
    min_side = min(a.shape[0], a.shape[1])
    all_samples = []
    for inset_div, patch_div in [(40, 80), (20, 40), (10, 20)]:
        patch = max(12, min_side // patch_div)
        inset = max(patch, min_side // inset_div)
        all_samples.append(_corner_block_sample(a, inset, patch))
    for t in (max(2, min_side // 200), max(4, min_side // 100), max(8, min_side // 50)):
        all_samples.append(_edge_ring_sample(a, t))

    votes = {}
    for sample in all_samples:
        cand = _dominant_color(sample)
        if cand is None:
            continue
        fam = _classify_family(cand)
        if fam is None:
            continue
        votes.setdefault(fam, []).append(cand)
    if not votes:
        return None
    best_family = max(votes, key=lambda f: len(votes[f]))
    bg = tuple(int(round(x)) for x in np.mean(votes[best_family], axis=0))
    return best_family, bg


# 依序嘗試的方法清單。新踩坑要新增方法時，加在這個list最後面，不要改動前面
# 已經驗證過的方法本體——這樣舊踩坑不會因為修新踩坑而復發。
BG_CANDIDATE_METHODS = [
    ("①grid_anchored(09-13,解799)", _bg_candidate_grid_anchored),
    ("②corner_shallow(09-12,解786/787)", _bg_candidate_corner_shallow),
    ("③edge_ring(早期方法)", _bg_candidate_edge_ring),
    ("④majority_all(最原始兜底)", _bg_candidate_majority_all),
]


def detect_bg_color(img_rgb):
    """回傳 (家族, 實際量到的底色RGB)。底色一律用「這張圖真的量到什麼顏色」，
    不snap到任何寫死的參考色，容忍背景本身沒有剛好是整數色號的情況。

    🔴2026-09-13重構：依序嘗試 BG_CANDIDATE_METHODS 裡累積的每一種方法，
    前一個方法「完全量不到底色」(結構性失敗，回傳None)才輪到下一個，量到
    就直接採用——不會有「兩個方法都算出結果、選哪個」的情況，因為實測證實
    這種情況下無法只憑像素統計特徵自動判斷哪個候選才正確(詳見上方架構說明
    的失敗紀錄：多方法投票、去背比例變異量兩種自動驗證法都試過，均不可靠)。
    方法①(幾何格線)在目前所有測過的真實圖與合成案例中沒有已知的錯誤案例，
    平時幾乎都在①就拿到答案；②③④只在①抓不到幾何格線的特殊圖才會用到，
    保留下來是避免那種情況直接報錯、完全處理不了。

    每個方法各自的踩坑背景，見各自函式的docstring；799(方法①解決)、786/787
    (方法②解決)分別是最近兩次的真實案例。"""
    for name, fn in BG_CANDIDATE_METHODS:
        cand = fn(img_rgb)
        if cand is not None:
            log(f"      [底色偵測] 方法 {name} 判斷結果：{cand}")
            return cand
    raise RuntimeError("多重偵測皆找不到明顯底色，圖片可能異常，請人工檢查")


def color_close(a, target, tol=40):
    return int(abs(int(a[0]) - target[0]) + abs(int(a[1]) - target[1]) + abs(int(a[2]) - target[2])) < tol


def scan_trim(img_arr, y0, y1, x0, x1, side, bgcolor):
    h, w, _ = img_arr.shape
    need_consecutive = 5
    if side in ("top", "bottom"):
        xs = range(max(x0, 0), min(x1, w), max(1, (x1 - x0) // 40))
        for depth in range(0, MAX_SIDE_TRIM):
            y = y0 + depth if side == "top" else y1 - 1 - depth
            if y < 0 or y >= h:
                break
            hits, samples = 0, 0
            for x in xs:
                samples += 1
                if color_close(img_arr[y, x], bgcolor):
                    hits += 1
            if samples and hits / samples > 0.9:
                ok = True
                for k in range(1, need_consecutive):
                    yy = y + k if side == "top" else y - k
                    if yy < 0 or yy >= h:
                        ok = False
                        break
                    row_hits = sum(1 for x in xs if color_close(img_arr[yy, x], bgcolor))
                    if row_hits / max(1, samples) < 0.85:
                        ok = False
                        break
                if ok:
                    return depth
        return MAX_SIDE_TRIM // 2
    else:
        ys = range(max(y0, 0), min(y1, h), max(1, (y1 - y0) // 40))
        for depth in range(0, MAX_SIDE_TRIM):
            x = x0 + depth if side == "left" else x1 - 1 - depth
            if x < 0 or x >= w:
                break
            hits, samples = 0, 0
            for y in ys:
                samples += 1
                if color_close(img_arr[y, x], bgcolor):
                    hits += 1
            if samples and hits / samples > 0.9:
                ok = True
                for k in range(1, need_consecutive):
                    xx = x + k if side == "left" else x - k
                    if xx < 0 or xx >= w:
                        ok = False
                        break
                    col_hits = sum(1 for y in ys if color_close(img_arr[y, xx], bgcolor))
                    if col_hits / max(1, samples) < 0.85:
                        ok = False
                        break
                if ok:
                    return depth
        return MAX_SIDE_TRIM // 2


def cover_wm(img, fill):
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([w - 40, h - 40, w, h], fill=fill)
    return img


# ---- 🔴 幾何格線偵測（2026-08-23 新增）--------------------------------
# 白色格線是「貫穿整張圖的直線」，是幾何特徵，跟角色多大、格內有沒有場景完全無關。
# 舊做法 scan_trim 是逐格掃底色，角色一大就掃不到「90% 是底色」的行，
# 直接 fallback 到 MAX_SIDE_TRIM//2=45 盲猜 → 該切 51 只切 45 就殘留白線，
# 該切 10 卻切 45 就啃掉角色。幾何法沒有這個問題。
GRID_WHITE_TH   = 235    # 判定白的最低亮度（RGB 三通道都要 >= 這個值）
GRID_MIN_RATIO  = 0.80   # 一整欄/列要有多少比例是白，才算格線
GRID_MIN_WIDTH  = 3      # 格線最小寬度(px)，比這細的當雜訊


def _white_runs(ratio, min_ratio=GRID_MIN_RATIO, min_width=GRID_MIN_WIDTH):
    on = ratio >= min_ratio
    runs, st = [], None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        elif not v and st is not None:
            runs.append((st, i - 1)); st = None
    if st is not None:
        runs.append((st, len(on) - 1))
    return [r for r in runs if r[1] - r[0] + 1 >= min_width]


def detect_grid(img_rgb, mode="green"):
    """靠格線的幾何位置切格。成功回傳 (xs, ys)：各 4 段 (start,end)；失敗回傳 None。
    綠/藍底的格線是白色；白底格線是黑色（白底畫白線會看不見），依 mode 換一套判斷方向，
    否則白底套組永遠找不到白色格線，會整批退回精準度較差的掃底色模式。"""
    a = np.array(img_rgb)
    h, w, _ = a.shape
    # 只有綠底用白色格線；其餘家族(藍/白/洋紅/灰)一律用黑色格線——同色系底配同色系線會看不見，
    # 之後新增的底色家族預設也走黑線，不用每加一種就回來改這裡。
    if mode == "green":
        line = (a.min(axis=2) >= GRID_WHITE_TH)
    else:
        line = (a.max(axis=2) <= NEAR_BLACK)
    xs = _white_runs(line.mean(axis=0))
    ys = _white_runs(line.mean(axis=1))

    def pad_to_four(runs, dim):
        # 原本假設整張圖外圈也框了一圈格線色，抓到的run會剛好是4段。
        # 但有些畫風(例如白底黑線)只有格子「中間」2條分隔線、外圍沒有畫框，
        # 這種情況真實抓到的只會有2段，要手動補上左右(或上下)兩個虛擬邊界，
        # 不然這種畫風永遠會落到len(runs)!=4被判定失敗，白白浪費掉已經抓對的2條線。
        if len(runs) == 4:
            return runs
        if len(runs) == 2:
            return [(-1, -1), runs[0], runs[1], (dim, dim)]
        return runs

    xs = pad_to_four(xs, w)
    ys = pad_to_four(ys, h)
    if len(xs) != 4 or len(ys) != 4:
        return None
    # 合理性檢查：三欄/三列寬度要接近（差距 < 12%），否則判定偵測失敗
    cw = [xs[i + 1][0] - xs[i][1] for i in range(3)]
    chh = [ys[i + 1][0] - ys[i][1] for i in range(3)]
    for arr in (cw, chh):
        if min(arr) <= 0 or (max(arr) - min(arr)) / max(arr) > 0.12:
            return None
    return xs, ys


def auto_crop_cells(img_rgb, bgcolor, mode="green"):
    # ① 先試幾何格線
    g = detect_grid(img_rgb, mode)
    if g:
        xs, ys = g
        h, w, _ = np.array(img_rgb).shape
        # 🔴2026-09-12修正（777殘留線）：格線偵測到的是「夠深/夠白」的核心，
        # 兩側還有肉眼看不明顯、但debg去背抓不到的反鋸齒漸層(量測777實際圖：
        # 乾淨底色跟線之間還有約8px的漸層地帶，SAFETY_BUFFER=3不夠蓋過去，
        # 切格後那圈漸層留在裁切範圍內，因為顏色不夠像任何一種底色家族，
        # debg永遠不會把它判定成可去背，變成飄在透明背景中間的一條細線)。
        # 改成跟畫布尺寸成比例的邊界，2048px量到的安全值約12px，用同比例套用到
        # 其他解析度；不影響下面②掃底色模式的SAFETY_BUFFER(那條路徑的裁切深度
        # 是實測掃出來的，不是固定核心區，本來就已經含了漸層)。
        line_margin = max(SAFETY_BUFFER, min(h, w) // 170)
        cells = []
        for r in range(3):
            y0 = ys[r][1] + 1 + line_margin
            y1 = ys[r + 1][0] - line_margin
            for c in range(3):
                x0 = xs[c][1] + 1 + line_margin
                x1 = xs[c + 1][0] - line_margin
                cells.append(img_rgb.crop((x0, y0, x1, y1)))
        print("      [切格] 幾何格線偵測成功（不受角色大小影響）")
        return cells

    # ② 退回逐格掃底色；找不到時不再盲猜 45，改抄同類型邊的中位數
    print("      [切格] 找不到白色格線，退回掃底色模式")
    a = np.array(img_rgb)
    h, w, _ = a.shape
    cw, ch = w // 3, h // 3
    FB = MAX_SIDE_TRIM // 2
    SIDES = ("top", "bottom", "left", "right")

    def is_outer(r, c, side):
        return ((side == "top" and r == 0) or (side == "bottom" and r == 2) or
                (side == "left" and c == 0) or (side == "right" and c == 2))

    raw = {}
    for r in range(3):
        for c in range(3):
            box = (r * ch, (r + 1) * ch, c * cw, (c + 1) * cw)
            raw[(r, c)] = {s: scan_trim(a, box[0], box[1], box[2], box[3], s, bgcolor) for s in SIDES}

    pool = {}
    for (r, c), d in raw.items():
        for s, v in d.items():
            if v != FB:
                pool.setdefault((s, is_outer(r, c, s)), []).append(v)
    # 同一邊沒樣本時，退而求其次用「同為外框 / 同為格線」的全部樣本
    wide = {}
    for (s, o), lst in pool.items():
        wide.setdefault(o, []).extend(lst)

    fixed = 0
    for (r, c), d in raw.items():
        for s in SIDES:
            if d[s] == FB:
                o = is_outer(r, c, s)
                cand = pool.get((s, o)) or wide.get(o)
                if cand:
                    d[s] = int(np.median(cand)); fixed += 1
    if fixed:
        print(f"      [切格] {fixed} 個邊掃不到底色，已改用同類型邊的中位數（原本會盲猜 {FB}）")

    cells = []
    for r in range(3):
        for c in range(3):
            d = raw[(r, c)]
            x0 = c * cw + d["left"] + SAFETY_BUFFER
            x1 = (c + 1) * cw - d["right"] - SAFETY_BUFFER
            y0 = r * ch + d["top"] + SAFETY_BUFFER
            y1 = (r + 1) * ch - d["bottom"] - SAFETY_BUFFER
            cells.append(img_rgb.crop((x0, y0, x1, y1)))
    return cells


def _bg_removable_mask(r, g, b, mode, bgcolor):
    """依家族＋這張圖實際量到的底色，判斷哪些pixel算底色可去除。
    彩色家族(綠/藍/洋紅)用色相關係判斷，容忍同一家族內深淺不一；
    無彩色家族(白/灰)用「跟量到的底色亮度夠接近」判斷，容忍白底不是剛好255。"""
    if mode == "green":
        return (r < 80) & (b < 80) & ((g - np.maximum(r, b)) > 40) & (g > 95)
    if mode == "blue":
        return (r < 80) & (g < 80) & ((b - np.maximum(r, g)) > 40)
    if mode == "magenta":
        return (g < 80) & ((np.minimum(r, b) - g) > 40)
    if mode in ("white", "gray"):
        bl = sum(bgcolor) / 3.0
        val = (r.astype(float) + g + b) / 3.0
        spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        return (spread < ACHROMATIC_SPREAD) & (np.abs(val - bl) < 40)
    return np.zeros(r.shape, dtype=bool)


def debg(cell, mode, bgcolor=(0, 128, 0)):
    im = cell.convert("RGBA")
    a = np.array(im)
    r = a[:, :, 0].astype(int)
    g = a[:, :, 1].astype(int)
    b = a[:, :, 2].astype(int)
    mask = _bg_removable_mask(r, g, b, mode, bgcolor)
    if not mask.any():
        a[:, :, 3] = 255
        return Image.fromarray(a)
    bgc = a[min(20, a.shape[0] - 1), min(20, a.shape[1] - 1), :3].astype(float)
    lbl, n = ndimage.label(mask)
    border = set(lbl[0, :]).union(lbl[-1, :]).union(lbl[:, 0]).union(lbl[:, -1])
    border.discard(0)
    keep = np.zeros(mask.shape, bool)
    for i in range(1, n + 1):
        comp = lbl == i
        if i in border:
            keep |= comp
            continue
        avg = a[comp][:, :3].astype(float).mean(axis=0)
        if np.abs(avg - bgc).sum() < 45:
            keep |= comp
    a[:, :, 3] = np.where(keep, 0, 255)
    return Image.fromarray(a)


def dehalo(im, mode, bgcolor=(0, 128, 0), passes=45):
    a = np.array(im)
    r = a[:, :, 0].astype(int)
    g = a[:, :, 1].astype(int)
    b = a[:, :, 2].astype(int)
    st = np.ones((3, 3), bool)
    if mode == "green":
        peelable = (r < 90) & (b < 90) & ((g - np.maximum(r, b)) > 25)
    elif mode == "blue":
        peelable = (r < 90) & (g < 90) & ((b - np.maximum(r, g)) > 25)
    elif mode == "magenta":
        peelable = (g < 90) & ((np.minimum(r, b) - g) > 25)
    elif mode in ("white", "gray"):
        bl = sum(bgcolor) / 3.0
        val = (r.astype(float) + g + b) / 3.0
        spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        peelable = (spread < ACHROMATIC_SPREAD) & (np.abs(val - bl) < 30)
    else:
        peelable = np.zeros(r.shape, dtype=bool)
    for _ in range(passes):
        al = a[:, :, 3]
        transp = al == 0
        td = ndimage.binary_dilation(transp, structure=st, iterations=1)
        rm = (al > 0) & td & peelable
        if not rm.any():
            break
        a[:, :, 3][rm] = 0
    return Image.fromarray(a)


def proc_cell_line(cell, size, mode, bgcolor=(0, 128, 0)):
    """LINE規格：去背+去白邊後，縮成固定小尺寸置中"""
    c = debg(cell, mode, bgcolor)
    c = dehalo(c, mode, bgcolor)
    c.thumbnail(size, Image.LANCZOS)
    cv = Image.new("RGBA", size, (0, 0, 0, 0))
    cv.paste(c, ((size[0] - c.width) // 2, (size[1] - c.height) // 2), c)
    return cv


def hole_report(out_dir):
    findings = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".png"):
            continue
        a = np.array(Image.open(os.path.join(out_dir, f)).convert("RGBA"))
        transp = a[:, :, 3] == 0
        lbl, n = ndimage.label(transp)
        border = set(lbl[0, :]).union(lbl[-1, :]).union(lbl[:, 0]).union(lbl[:, -1])
        for i in range(1, n + 1):
            if i in border:
                continue
            size = int((lbl == i).sum())
            if size > HOLE_REPORT_PX:
                findings.append((f, size))
    return findings


def build_preview(out_dir, preview_path):
    fs = [f"{i:02d}.png" for i in range(1, 41)] + ["main.png", "tab.png"]
    cols, cell = 8, 200
    rows = (len(fs) + cols - 1) // cols
    cv = Image.new("RGB", (cols * cell, rows * cell), (255, 0, 255))
    for i, f in enumerate(fs):
        p = os.path.join(out_dir, f)
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGBA")
        im.thumbnail((cell - 10, cell - 10))
        cv.paste(im, ((i % cols) * cell + (cell - im.width) // 2,
                       (i // cols) * cell + (cell - im.height) // 2), im)
    cv.save(preview_path)


# ================================================================
# 第二部分：Redbubble專用 —— 原生解析度 + 清雜點 + 留白邊界
# ================================================================

def remove_stray_specks(im):
    """清掉跟角色本體不相連、面積很小的雜散色塊(機器裁切容易誤判的雜訊)"""
    a = np.array(im)
    alpha = a[:, :, 3]
    opaque = alpha > 0
    lbl, n = ndimage.label(opaque)
    if n <= 1:
        return im, n
    sizes = ndimage.sum(opaque, lbl, range(1, n + 1))
    main_id = int(np.argmax(sizes)) + 1
    remaining = 1
    for i in range(1, n + 1):
        if i == main_id:
            continue
        if sizes[i - 1] < RB_STRAY_MIN_PX:
            a[:, :, 3][lbl == i] = 0
        else:
            remaining += 1  # 面積不小的獨立色塊(如手持道具)，保留但列入需人工確認名單
    return Image.fromarray(a), remaining


def trim_to_content(im):
    """裁到角色實際不透明範圍的最小外框"""
    a = np.array(im)
    alpha = a[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return im
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return im.crop((x0, y0, x1, y1))


def add_peel_margin(im, margin_ratio=RB_MARGIN_RATIO):
    """角色置中，四周補透明留白 → 輸出正方形畫布(對應Redbubble撕除背膠的留白要求)"""
    w, h = im.size
    long_side = max(w, h)
    margin = int(round(long_side * margin_ratio))
    canvas_side = long_side + margin * 2
    cv = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    cv.paste(im, ((canvas_side - w) // 2, (canvas_side - h) // 2), im)
    return cv


def proc_cell_redbubble(cell, mode, bgcolor=(0, 128, 0)):
    """Redbubble規格：去背+去白邊+清雜點+裁到內容+補留白，不縮小、保留原生解析度"""
    c = debg(cell, mode, bgcolor)
    c = dehalo(c, mode, bgcolor)
    c, remaining = remove_stray_specks(c)
    c = trim_to_content(c)
    c = add_peel_margin(c)
    return c, (remaining > 1)



# ================================================================
# ★新增：素材尋找 + 有字/無字自動判斷 + 疊字
# ================================================================

def find_src(folder, setn, i):
    """找 {編號}-{i}，副檔名不限。回傳完整路徑，找不到回傳 None。"""
    for ext in IMG_EXT:
        p = os.path.join(folder, f"{setn}-{i}{ext}")
        if os.path.exists(p):
            return p
    return None


def find_texts(setn):
    """🔴有字／無字看檔案自動判斷，回傳 40 個文案的 list（不疊字回傳 None）。

    找的順序：
      ① 文本/{編號}_plan.json    [["文案","動畫類型"], ...]  → 只取文案
      ② 文本/{編號}_文案.json    ["文案", ...] 或 {"01":"文案", ...}
    一個都沒有 → None（＝Gemini 圖上已自帶文字，不用再疊）
    """
    if NO_TEXT:
        return None

    def pick(d):
        for base in (TEXT_DIR, HERE, ROOT):
            p = os.path.join(base, d)
            if os.path.exists(p):
                return p
        return None

    def to40(v):
        """統一轉成長度 40 的 list，不足補空字串。"""
        if isinstance(v, dict):
            v = [v.get(f"{i:02d}") or "" for i in range(1, 41)]
        v = [x[0] if isinstance(x, (list, tuple)) else (x or "") for x in v]
        return (list(v) + [""] * 40)[:40]

    p = pick(f"{setn}_plan.json")
    if p:
        log(f"[T] 有字版：{os.path.basename(p)}")
        return to40(json.load(open(p, encoding="utf-8")))

    p = pick(f"{setn}_文案.json")
    if p:
        log(f"[T] 有字版：{os.path.basename(p)}")
        return to40(json.load(open(p, encoding="utf-8")))

    log("[T] 無字版（找不到文案檔，圖上文字沿用原圖）")
    return None


_FONT_CACHE = {}


def load_font(size):
    if size not in _FONT_CACHE:
        f = None
        for path in (FONT_PATH, FONT_ALT):
            try:
                f = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text(canvas, text):
    """黑字白描邊，水平置中畫在畫布上方。字太寬會自動縮到塞得下。
    """
    if not text:
        return canvas
    size = FONT_SIZE
    while size > 20:
        font = load_font(size)
        d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bb = d.textbbox((0, 0), text, font=font, stroke_width=FONT_STROKE)
        if bb[2] - bb[0] <= canvas.width - 16:
            break
        size -= 3
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = (canvas.width - (bb[2] - bb[0])) // 2 - bb[0]
    y = TEXT_TOP - bb[1]
    d.text((x, y), text, font=font, fill=(0, 0, 0, 255),
           stroke_width=FONT_STROKE, stroke_fill=(255, 255, 255, 255))
    return Image.alpha_composite(canvas, layer)


def fit_below_text(im, text):
    """有字時角色要往下讓出文字空間，不然頭會被字蓋到。
    無字就原樣回傳。"""
    if not text:
        return im
    font = load_font(FONT_SIZE)
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bb = d.textbbox((0, 0), text, font=font, stroke_width=FONT_STROKE)
    top_pad = TEXT_TOP + (bb[3] - bb[1]) + 10          # 文字高 + 間距
    W, H = im.size
    ch = im.getbbox()
    if not ch:
        return im
    body = im.crop(ch)
    avail_h = H - top_pad - 6
    r = min(1.0, avail_h / body.height, (W - 16) / body.width)
    body = body.resize((max(1, round(body.width * r)),
                        max(1, round(body.height * r))), Image.LANCZOS)
    c = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    c.paste(body, ((W - body.width) // 2, H - 6 - body.height), body)   # 靠底＝腳踩地
    return c


# ================================================================
# 第三部分：掃描 + 合併主流程
# ================================================================

def find_all_ready_sets(folder):
    if not os.path.isdir(folder):
        log(f"❌ 找不到資料夾: {folder}")
        sys.exit(1)
    names = os.listdir(folder)
    pat = re.compile(r"^(\d+)-([1-5])$")
    groups = {}
    for n in names:
        stem, ext = os.path.splitext(n)
        if ext.lower() not in IMG_EXT:
            continue
        m = pat.match(stem)
        if m:
            setn, idx = m.group(1), int(m.group(2))
            groups.setdefault(setn, set()).add(idx)
    ready = []
    for setn, idxs in sorted(groups.items(), key=lambda x: int(x[0])):
        if idxs == {1, 2, 3, 4, 5}:
            ready.append(setn)
        else:
            log(f"⚠ 編號 {setn} 只有 {sorted(idxs)}，缺圖，暫不處理")
    return ready


def line_output_done(folder, setn):
    return os.path.exists(os.path.join(folder, f"{setn}_output.zip"))


def redbubble_output_done(setn):
    d = os.path.join(RB_OUT_ROOT, setn)
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def process_one(folder, setn):
    need_line = not line_output_done(folder, setn)
    need_rb = not redbubble_output_done(setn)
    if not need_line and not need_rb:
        log(f"= {setn} LINE與Redbubble皆已完成，跳過 =")
        return

    files = [find_src(folder, setn, i) for i in range(1, 6)]
    if any(f is None for f in files):
        raise FileNotFoundError(f"{setn} 缺圖（1~5 沒齊）")
    log(f"=== 處理 {setn}  (LINE:{'需做' if need_line else '已完成'} / Redbubble:{'需做' if need_rb else '已完成'}) ===")
    texts = find_texts(setn)
    first_img = Image.open(files[0]).convert("RGB")
    mode, bgcolor = detect_bg_color(first_img)
    fillcolor = bgcolor  # 蓋Gemini浮水印一律用「這張圖實際量到的底色」，不用寫死對照表
    log(f"[1] 自動偵測底色: {mode} (RGB={bgcolor})"
        f"{'　【疊字】' if texts else '　【不疊字】'}")

    out_dir = os.path.join(folder, f"{setn}_output")
    rb_dir = os.path.join(RB_OUT_ROOT, setn)
    if need_line:
        os.makedirs(out_dir, exist_ok=True)
    if need_rb:
        os.makedirs(rb_dir, exist_ok=True)

    log("[2] 逐格裁切中(LINE小圖與Redbubble原生大圖共用同一次裁切結果)...")
    num, rb_num = 1, 1
    rb_warn = []
    for gi, fp in enumerate(files):
        g = gi + 1
        img = Image.open(fp).convert("RGB")
        img = cover_wm(img, fillcolor)
        cells = auto_crop_cells(img, bgcolor, mode)
        for ci in range(9):
            is_special = (ci == 8)

            if need_line:
                if is_special:
                    if g == 4:
                        proc_cell_line(cells[ci], MAIN, mode, bgcolor).save(
                            os.path.join(out_dir, "main.png"), "PNG", optimize=True)
                    elif g == 5:
                        proc_cell_line(cells[ci], TAB, mode, bgcolor).save(
                            os.path.join(out_dir, "tab.png"), "PNG", optimize=True)
                else:
                    cell = proc_cell_line(cells[ci], STK, mode, bgcolor)
                    if texts:
                        txt = texts[num - 1] if num <= len(texts) else ""
                        cell = draw_text(fit_below_text(cell, txt), txt)
                    cell.save(os.path.join(out_dir, f"{num:02d}.png"),
                              "PNG", optimize=True)

            if need_rb:
                if not (is_special and g not in (4, 5)):
                    final, flag = proc_cell_redbubble(cells[ci], mode, bgcolor)
                    tag = "main" if (is_special and g == 4) else ("tab" if (is_special and g == 5) else f"{rb_num:02d}")
                    out_path = os.path.join(rb_dir, f"{setn}_{tag}.png")
                    final.save(out_path, "PNG", optimize=True)
                    if flag:
                        rb_warn.append(out_path)

            if not is_special:
                num += 1
                rb_num += 1

    if need_line:
        log("[3] 打包LINE用ZIP + 洋紅預覽...")
        zip_path = os.path.join(folder, f"{setn}_output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(out_dir)):
                zf.write(os.path.join(out_dir, f), f)
        preview_path = os.path.join(folder, f"{setn}_preview.png")
        build_preview(out_dir, preview_path)
        findings = hole_report(out_dir)
        if findings:
            for f, sz in findings:
                log(f"    ⚠ {f} 鏤空面積={sz}px → 請人工確認是否為合法設計")
        log(f"    ✅ LINE輸出完成: {zip_path}")

    if need_rb:
        log(f"    ✅ Redbubble輸出完成: {rb_dir}  共{rb_num - 1}張+main+tab")
        if rb_warn:
            log("    ⚠ 以下檔案清完雜點後仍有多個不相連大色塊，麻煩人工確認是否合法(如手持道具)：")
            for p in rb_warn:
                log(f"       {p}")
    log("")


def main():
    args = sys.argv[1:]
    folder = INPUT_DIR
    if args:
        setns = args
    else:
        log(f"掃描資料夾: {folder}")
        setns = find_all_ready_sets(folder)
        if not setns:
            log("目前沒有5張圖齊全的編號可處理。")
            return
        log(f"待檢查編號: {setns}")
        log("")
    for setn in setns:
        try:
            process_one(folder, setn)
        except Exception as e:
            log(f"❌ 編號 {setn} 處理失敗: {e}")
            log("")
    log("=== 全部完成 ===")


def _pause():
    """雙擊執行時視窗不要瞬間消失。"""
    try:
        input("\n按 Enter 關閉…")
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _pause()
