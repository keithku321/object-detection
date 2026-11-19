# Proposal Outline (Draft for Review)

Status: outline only. The full English proposal will be written after this outline is confirmed.

Sources: *Topic 4 / Topic 5 – Writing Project Proposal* (Parts 1 and 2), and the plans in `plans/` (A to D, main plan, team workflow, decisions log) plus the state described in `CLAUDE.md`.

---

## 0. 先要決定嘅幾件事（會影響成份 proposal）

1. **一份 proposal 寫晒 4 個 plan。** 指引講明「Problems too many, too broad = lack of project aim」，所以我哋要用**一個統一嘅問題**包住四個子題目（見 1.2）。
2. **字數：around 3,000 words。** 四個子題目要塞入 3,000 字，所以每個子題目大約得 150 至 250 字。圖表同 appendix 預設唔計入字數（要同導師確認）。
3. **Plan D（CamoEdge）唔係 YOLO。** 我會寫成「同一個 benchmark、同一個 demo 之下，針對『偽裝』呢個視覺條件嘅專門模型」，同時喺 Section 2.2 老實講佢同其他三個嘅分別。
4. **Preliminary work 只寫真正做過嘅嘢。** 目前只有 Plan A 喺真實數據（Brackish）訓練過（1 epoch 同 20 epoch）；B、C、D 只喺 tiny data / coco8 跑過 pipeline。Proposal 會照直寫，唔會誇大。

---

## Cover Page（唔計字數）

| Item | 內容 | 狀態 |
|---|---|---|
| Name（4 位組員） | 待你提供 | 需要你提供 |
| HKMU ID | 待你提供 | 需要你提供 |
| Programme of Study | Computer Science（CLAUDE.md 提過），要確認正式 programme 名稱同 code | 需確認 |
| Date | 提交日期 | 需要你提供 |
| Supervisor | Plan A 文件提到 Dr. Wang Weiming Wyman，要確認係咪成組嘅 supervisor | 需確認 |
| Project Title | 要同 nomination stage 嘅題目**不同**；一句、描述性、精簡 | 見下面候選 |

候選題目（指引：列出 problem domain、technology、technique、approach 嘅名詞，再排序）：

- *Robust Lightweight Object Detection under Challenging Visual Conditions: Underwater, Adverse Weather, Small-Scale and Camouflaged Scenes*
- *Condition-Aware Lightweight Detectors: Attention, Distillation and Boundary Guidance for Robust Real-Time Detection*
- *Making Lightweight Detectors See Clearly in Water, Weather, Distance and Disguise*（比較口語，唔太建議）

---

## Table of Contents（唔計字數）

按指引 Part 2 嘅 template：Chapter 1 至 3、References、Appendix A 至 C。

---

## Chapter 1. Introduction (Problem Definition) — 目標約 750 words

### 1.1 Introduction (Why / background) — 約 250 words
- Object detection 喺受控環境已經成熟；真實部署（海底機械人、交通監察、無人機、野外監測）會遇到劣化條件。
- 定義四個條件同佢哋點樣令 lightweight detector 失準：
  - 水下：散射同色偏令對比下降（Brackish、URPC）
  - 惡劣天氣：霧、雨、雪、沙（DAWN）
  - 小物件：航拍下物件好細，downsample 之後消失（VisDrone）
  - 偽裝物件：前景同背景紋理相近，邊界模糊（COD10K、CAMO）
- 共同矛盾：**準確度要升，但模型要細、要快**（邊緣部署、≥ 30 FPS）。
- 點出 gap：通常每個條件各自獨立研究，缺乏統一協議（同一 split 原則、3 seeds、同一個速度測量方法）去公平比較。

### 1.2 Project Aim and Objectives — 約 350 words
- **Aim（一句，聚焦）：** 設計、實現同評估一組針對四種挑戰視覺條件嘅輕量化偵測模型，並用統一協議比較，最後整合入同一個 demo。
- 解釋 aim 入面嘅關鍵詞（robust、lightweight、challenging visual conditions）。
- **Objectives（目標 7 個，指引建議 5 至 7 個，要可量度、用 action verbs，最少一個係 evaluation）：**

| # | Objective（草稿） | 對應 plan | 可量度嘅完成標準 |
|---|---|---|---|
| O1 | Setup 統一資料、評估同紀錄基礎設施（manifest + SHA-256、固定 split、results CSV、speed benchmark） | 全部 | 四個任務各自有 baseline 結果列 |
| O2 | Design and implement 輕量 attention 模組，加入 YOLO26 做水下偵測 | A | 同 size baseline 之上提升 mAP50（3 seeds，mean ± std）；≥ 30 FPS |
| O3 | Design and implement 知識蒸餾（feature + class + box），YOLO26m → YOLO26n | B | student 勝過 plain student；參數量不變；有分天氣結果 |
| O4 | Design and implement 小物件偵測改良（P2 head、attention、NWD loss） | C | AP_small 高過 baseline；整體 mAP 無崩 |
| O5 | Design and implement CamoEdge（邊界引導偽裝物件分割） | D | S_α、E_φ、F_β^w 上升、MAE 下降；有 boundary metric |
| O6 | Implement 一個 Gradio demo，每個任務一個 tab（圖片 / 影片 / webcam） | 全部 | 四個 tab 都可以跑 |
| O7 | Evaluate 全部模型：消融實驗、多 seeds、accuracy–size–speed 表、失敗個案分析 | 全部 | 報告所需數字全部有 mean ± std |

- 會畫一張 Problem → Aim → Objectives 圖（跟指引嘅 elevator 例子），顯示 O1 係基礎、O2 至 O5 並行、O6 整合、O7 評估。

### 1.3 Value Proposition — 約 150 words
- **Immediate value：** 四個條件各有一個可重現嘅輕量模型；每個設計決定都有消融證據；一套可重用嘅統一評估協議。
- **Ripple effect：** 對水下機械人、交通監察、無人機巡檢、野生動物 / 軍事偽裝偵測等應用有參考價值；開源 pipeline 方便日後換模型再比較；負面結果都會報告，對社群有價值。

---

## Chapter 2. Background / Literature Review — 目標約 950 words

指引重點：寫**經驗教訓（正面要採用、負面要避免）**，並以 domain requirements 作結；要用 requirements 評估現有方案嘅 suitability；引用要來自 journals / conference proceedings（最少 6 個）。

### 2.1 Review of Existing or Related Solutions — 約 700 words
| 小節 | 內容 | 約字數 |
|---|---|---|
| 2.1.1 Real-time detectors | YOLO 系列演進；YOLO26 嘅特點（無 DFL、NMS-free、STAL、ProgLoss）；點解揀 lightweight YOLO | 150 |
| 2.1.2 Underwater | 先增強後偵測 vs 直接喺模型內加 attention（CBAM、CA、ECA、SimAM）；Brackish、URPC / DUO；教訓：增強階段加延遲，URPC 有重複圖 | 150 |
| 2.1.3 Adverse weather | 去霧 / 去雨前處理 vs 蒸餾（Hinton 等）；DAWN；教訓：小數據集好易 overfit，要 5-fold | 150 |
| 2.1.4 Small objects | FPN / BiFPN、P2 head、NWD、SAHI 切片推理、DySample；VisDrone；教訓：YOLO26 STAL 已經吸收部份增益 | 150 |
| 2.1.5 Camouflaged objects | SINet / SINet-V2、邊界感知方法、PVTv2 encoder、COD10K / CAMO；教訓：全域指標會隱藏邊界進步 | 100 |
| **Domain requirements（小節結尾）** | 整理成一個表：輕量、≥ 30 FPS、對特定劣化穩健、公平比較、可重現 | — |

### 2.2 Highlight of the Proposed Solution — 約 250 words
- 對照現有方案，列明我哋嘅主要分別（一個比較表）。
- **老實講強弱：** 我哋嘅模型唔會贏過商業或最新 SOTA；邊度可以做得更好（同一 backbone 之上加輕量模組、統一協議），邊度做唔到（無 Jetson 實機測試、free Colab 算力有限、數據集細）。
- 明確講出 YOLO26 已含 STAL / C2PSA，所以預期增益可能細；Plan D 唔屬 YOLO 家族。

---

## Chapter 3. Preliminary Methodology — 目標約 950 words

### 3.1 Overview — 約 150 words
- 方法論總覽：divide and conquer（共用基礎 + 四個並行子系統）、同一個控制變因原則（baseline 同 variants 用相同 split、image size、epochs、augmentation）、Agile 式 phase gate（G0 至 G6）。
- 每個 objective 點樣各自運作同點樣合起上嚟。

### 3.2 Requirements, Supporting Technologies and Technical Gap — 約 400 words
按 Part 2 指引嘅表格格式：

| Element | 內容 |
|---|---|
| Use cases + user groups | 研究者（訓練、比較模型）、應用開發者（部署輕量模型）、示範觀眾（上載圖片 / 影片 / webcam） |
| Functional requirements | 下載與驗證數據；訓練 / 恢復 / 評估；寫結果紀錄；速度基準；各任務推理；Gradio demo |
| Non-functional requirements | ≥ 30 FPS（batch 1、FP16、640 px、end-to-end）；參數量 / GFLOPs；可重現（pinned Ultralytics 8.4.175、記錄 commit 同 GPU）；可恢復（free Colab） |
| Essential key technologies | PyTorch、Ultralytics YOLO26、timm（PVTv2）、pycocotools、PySODMetrics、torch-pruning / ONNX Runtime、Gradio、Colab、Hugging Face |
| Technical gap | 例如：自訂 layer 令 pretrained transfer 失效（要檢查 "Transferred N/M"）；YOLO26 冇 DFL 所以 distillation 要改成 box-level；NMS-free 上限 300 detections 對密集場景嘅影響 |

### 3.3 Architecture / High-Level System Design — 約 400 words（不計圖表）
指引要求**最少 2 張圖**，再用表格描述 components：

- **圖 1：Component diagram** — 共用層（`common/`：datasets、results、benchmark、modules、training、plotResults）→ 四個任務模組（`underwater/`、`weather/`、`smallobject/`、`camoedge/`）→ `demo/app.py`。
- **圖 2：Data-flow diagram** — 數據集 → 下載與驗證 → 轉換 / 切分 → 訓練（含各任務特有部份）→ 評估（mAP / AP_s / 分天氣 / S-measure）→ results CSV 同 plots → 匯出 → demo 推理。
- 再加四張細圖或一張合併圖：
  - A：attention 模組喺 backbone / neck 嘅插入位置
  - B：teacher–student 同三個蒸餾 loss
  - C：P2 head、DySample、加權融合、NWD
  - D：encoder → EGB → EAG → decoder
- Components 描述表（major components、connections、data flow、input / output），列出各組件嘅用途同技術。

---

## References — 至少 6 個，計劃用 15 至 20 個

- 格式：**APA**（指引示範過；亦可揀 IEEE，全篇統一）。
- 草稿候選（寫正式版前會逐一用搜尋核對作者、年份、場刊）：
  YOLO 原論文、CBAM、Coordinate Attention、ECA-Net、SimAM、Hinton 蒸餾、DAWN、VisDrone、Brackish、DUO、NWD、SAHI、BiFPN / EfficientDet、DySample、SINet / SINet-V2、COD10K、PVTv2。
- 數據集同 Ultralytics 文件會另外列為技術文件，唔當學術文獻。

---

## Appendix A. Project Plan（不計字數）

- 取材自 `plans/00-main-plan.md` 嘅 Phase 0 至 7 同 gate G0 至 G6。
- 格式：phase、task、負責人、預計週數、deliverable。
- **需要你提供：** FYP 實際開始同截止日期（plans 刻意冇寫週數）。

## Appendix B. Members' Roles and Responsibilities（不計字數）

- 角色按 Part 2：Team Coordinator、Database Expert、System Analyst and Designer、Programmer、Tester and Evaluator、Experiment Designer and Data Analyst；**所有組員都要分擔 Analyst/Designer、Programmer、Tester**。
- 另列每人負責嘅 plan（A 至 D）同負責嘅 shared files（steward 分工，見 `plans/05-team-workflow.md`）。
- **需要你提供：** 邊個負責邊個 plan（decision D7 仍然 open）。

## Appendix C. Meeting Minutes（選填）

- 只放**真實**開會紀錄（日期、出席者、決定、行動項目）。
- **需要你提供**；我唔會自己編寫。如果冇，可以略過。

---

## 字數預算（共約 3,000）

| 部份 | 字數 |
|---|---|
| Chapter 1 | 750 |
| Chapter 2 | 950 |
| Chapter 3 | 950 |
| 圖表標題、過渡句等 | 約 350 |
| **合計** | **約 3,000** |

---

## 需要你確認 / 提供

1. 四位組員姓名、HKMU ID、正式 programme 名稱、supervisor、提交日期。
2. 邊個負責邊個 plan（Appendix B）。
3. FYP 開始同截止日期（Appendix A）。
4. 字數限制係咪只計 Chapter 1 至 3；圖表同 appendix 係咪不計。
5. 引用格式想用 APA 定 IEEE。
6. 最後輸出格式：Word（.docx）、Google Doc、定 PDF。
7. 有冇真實 meeting minutes 想放入 Appendix C。
8. Repo 嘅 `LICENSE.txt` 而家係 AGPL v3，但 plans 仍然寫 MIT；proposal 入面唔會提 licence，除非你想寫。
