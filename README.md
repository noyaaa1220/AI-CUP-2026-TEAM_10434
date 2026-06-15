# AI CUP 2026 春季賽：基於時序資料之桌球戰術與結果預測競賽

本專案為 AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測競賽」之程式碼與實驗流程。任務目標是根據每個桌球 rally 已公開的前 `n-1` 拍資訊，預測第 `n` 拍的：

- `actionId`：下一拍球種
- `pointId`：下一拍落點
- `serverGetPoint`：該 rally 最終是否由發球方得分之機率

本專案最終採用 **V3 transductive** 方法，核心設計包含：

1. 將 rally 轉換為 `prefix transition` 訓練樣本
2. 使用 `single-task GRU` 分別訓練 `actionId`、`pointId`、`serverGetPoint`
3. 對 `actionId` 與 `pointId` 使用 `test_new.csv` 中已公開 prefix 內部的 transition 進行合法的 transductive adaptation
4. 結合神經網路模型與統計 `backoff` 模型進行機率融合
5. 對 `serverGetPoint` 採取保守訓練，避免過擬合

---

## 1. 專案檔案說明

建議 GitHub 專案結構如下：

```text
.
├── V3.py
├── README.md
├── train.csv
├── test_old.csv
├── test_new.csv
└── submission_v3_transductive.csv
```


---

## 2. 執行環境

本實驗於 Linux 工作站完成，主要環境如下：

| 類別 | 項目 / 套件 | 版本 |
|---|---|---|
| 作業系統 | Ubuntu Linux | 22.04 |
| 系統核心 | Linux kernel | 6.8.0-90-generic |
| 開發環境 | Miniconda | - |
| 程式語言 | Python | 3.13.13 |
| 深度學習框架 | PyTorch | 2.11.0 |
| 資料處理 | pandas | 3.0.2 |
| 數值運算 | NumPy | 2.4.3 |
| 視覺化 | matplotlib | - |
| GPU | NVIDIA GeForce RTX 3090 | - |

本研究未使用外部資料集與預訓練模型，所有模型皆由競賽官方提供資料訓練。

---

## 3. 安裝套件

建議使用 Conda 建立獨立環境：

```bash
conda create -n aicup2026 python=3.13 -y
conda activate aicup2026
```

安裝主要套件：

```bash
pip install numpy pandas matplotlib torch
```

若使用 CUDA / GPU，請依照本機 CUDA 版本安裝對應的 PyTorch 版本。

---

## 4. 資料準備

請將以下三個官方資料檔放在同一個資料夾中：

```text
train.csv
test_old.csv
test_new.csv
```

程式預設會從目前目錄讀取資料；若資料放在其他資料夾，可使用 `--data-dir` 指定路徑。

---

## 5. 方法概述

### 5.1 Prefix transition

本任務是根據前 `n-1` 拍預測第 `n` 拍。因此，本專案將每個 rally 轉換為多筆 `prefix transition` 樣本：

```text
前 1 拍 → 預測第 2 拍
前 2 拍 → 預測第 3 拍
前 3 拍 → 預測第 4 拍
...
前 t 拍 → 預測第 t+1 拍
```

每筆樣本包含：

- 已觀察到的前 `t` 拍序列
- 下一拍查詢特徵
- 下一拍 `actionId`
- 下一拍 `pointId`
- 該 rally 的 `serverGetPoint`，僅限有標籤資料

此設計可增加訓練樣本數，並讓訓練資料形式更接近最終測試情境。

---

### 5.2 Transductive adaptation

`test_new.csv` 中雖然沒有真正要預測的第 `n` 拍答案，但官方公開了每個 rally 的前 `n-1` 拍。因此，本專案在最終版本中，對 `actionId` 與 `pointId` 額外使用 `test_new.csv` 已公開 prefix 內部的 observed transition。

例如，若某筆 test_new rally 公開第 1 到第 5 拍，而真正要預測第 6 拍，程式只使用：

```text
第 1 拍 → 第 2 拍
第 1~2 拍 → 第 3 拍
第 1~3 拍 → 第 4 拍
第 1~4 拍 → 第 5 拍
```

不使用：

```text
第 1~5 拍 → 第 6 拍
```

也就是不使用 hidden target。此做法僅用於 `actionId` 與 `pointId`，`serverGetPoint` 因為 `test_new.csv` 沒有最終勝負標籤，因此不使用 test_new 進行 supervised training。

---

### 5.3 Single-task GRU 模型

最終版本將三個任務分開訓練：

| 任務 | 輸出 | Loss function | 設計重點 |
|---|---|---|---|
| actionId | 15 類 | CrossEntropyLoss | 處理球種類別不平衡 |
| pointId | 10 類 | CrossEntropyLoss | 處理落點類別不平衡 |
| serverGetPoint | 0~1 機率 | BCEWithLogitsLoss | 維持 AUC 排序能力並避免過擬合 |

模型主要包含：

1. **序列分支**：將已觀察前 `t` 拍的類別特徵輸入 embedding，與數值特徵串接後，送入 2-layer bidirectional GRU。
2. **查詢分支**：描述下一拍預測情境，例如 prefix 長度、下一拍擊球者、比分狀態、前幾拍球種與落點。
3. **特徵融合**：將序列表示與查詢表示串接，經 shared MLP 後輸出各任務預測。

---

### 5.4 Backoff 與機率融合

除了神經網路模型，本專案也建立統計 backoff 模型。Backoff 會根據下列條件 key 統計條件機率分布：

- `prefixLenBucket`
- `nextStrikeId`
- `lastActionId`
- `lastPointId`
- `prevActionId`
- `prevPointId`
- `spinId`
- `handId`
- `sex`
- `score`

若較細條件的樣本數不足，會回退到較粗的條件分布。

最終提交不是單純使用神經網路輸出，而是進行加權融合：

| 任務 | Neural 權重 | Backoff 權重 |
|---|---:|---:|
| actionId | 0.45 | 0.55 |
| pointId | 0.45 | 0.55 |
| serverGetPoint | 0.35 | 0.65 |

---

## 6. 執行方式

### 6.1 最終提交版本

若資料檔案與 `V3.py` 放在同一層目錄，可執行：

```bash
python V3.py \
  --require-gpu \
  --method ensemble \
  --model-version single-task \
  --use-test-observed-transitions \
  --output submission_v3_transductive.csv
```

若要指定資料資料夾：

```bash
python V3.py \
  --data-dir ./data \
  --require-gpu \
  --method ensemble \
  --model-version single-task \
  --use-test-observed-transitions \
  --output submission_v3_transductive.csv
```

執行完成後會產生：

```text
submission_v3_transductive.csv
```

輸出欄位包含：

```text
rally_uid, actionId, pointId, serverGetPoint
```

---

### 6.2 CPU 測試執行

若只是想確認程式流程是否能跑，可以使用 CPU：

```bash
python V3.py \
  --cpu \
  --method ensemble \
  --model-version single-task \
  --use-test-observed-transitions \
  --output submission_v3_transductive.csv
```

注意：CPU 執行會明顯較慢。

---

### 6.3 Validation 模式

若要在本地進行 validation，可使用：

```bash
python V3.py \
  --require-gpu \
  --validate \
  --method ensemble \
  --model-version single-task
```

程式會輸出：

- backoff validation score
- neural validation score
- best blend validation setting

---

## 7. 主要參數說明

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--method` | `ensemble` | 使用 `backoff`、`neural` 或 `ensemble` |
| `--model-version` | `single-task` | 使用 `single-task` 或 `multitask` |
| `--use-test-observed-transitions` | `False` | 啟用後使用 test_new 已公開 prefix transition 訓練 action / point |
| `--seeds` | `42 2026 777` | 多 seed 模型平均 |
| `--epochs` | `28` | action / point 預設訓練 epoch |
| `--server-epochs` | `1` | server 任務訓練 epoch |
| `--batch-size` | `1024` | batch size |
| `--hidden` | `128` | GRU hidden size |
| `--dropout` | `0.18` | dropout rate |
| `--lr` | `0.002` | learning rate |
| `--weight-decay` | `0.0001` | AdamW weight decay |
| `--label-smoothing` | `0.02` | action / point label smoothing |
| `--action-nn-weight` | `0.45` | action neural probability 融合權重 |
| `--point-nn-weight` | `0.45` | point neural probability 融合權重 |
| `--server-nn-weight` | `0.35` | server neural probability 融合權重 |
| `--action-prior-adjust` | `0.35` | action prior adjustment 強度 |
| `--point-prior-adjust` | `0.40` | point prior adjustment 強度 |

---

## 8. 實驗結果

最終採用版本為 **V3 transductive**。

| 版本 | 模型架構 | Validation | Public | Private |
|---|---|---:|---:|---:|
| V1 | multi-task GRU + backoff | 0.335718 | 0.4444661 | - |
| V2 | action / point / server single-task + backoff | 0.338474 | 0.4433851 | - |
| V3 | V2 + 上下文特徵與 backoff 融合 | 0.352741 | 0.4440464 | - |
| V3 transductive | V3 + test_new 已公開 prefix transition | 0.352741 | 0.4464793 | 0.3774274 |
| V4 | 移除部分上下文特徵並加入保守分布修正 | 0.359547 | 0.4426977 | - |

由結果可見，V3 transductive 在 public leaderboard 上取得最高分，顯示 test_new 已公開 prefix transition 對 `actionId` 與 `pointId` 的測試分布適應具有幫助。

---

## 9. 注意事項

1. `test_new.csv` 只用於 actionId / pointId 的公開 prefix 內部 transition。
2. 不使用真正需要預測的第 `n` 拍 hidden target。
3. `serverGetPoint` 不使用 test_new 進行 supervised training。
4. 最終提交結果由模型推論與機率融合產生，未使用人工修正。
5. 若使用 `--require-gpu`，請確認 CUDA 與 NVIDIA driver 可正常使用。

