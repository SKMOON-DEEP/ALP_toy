## TL;DR (상사/기획자 보고용 한 장 요약)

* **“Form”의 의미**는 문서 양식이 아니라, **각 타깃 소리(예: 팬/베어링/밸브 등)마다 고객이 우리에게 제공해야 할 “정상음/현장 노이즈/메타데이터”의 최소·권장 수량(초/클립 수)**입니다.
* 현재 4개 스크립트는 **“고객 데이터 → 표준화/메타데이터 → 3가지 진단 알고리즘(A/B/C) → 점수/지표 출력”**을 *toy*로 end-to-end로 재현합니다.
* **A(Approach A)**: *학습 없이* (training-free) 정상 몇 개(3~10개)만으로 바로 동작. Frozen AST + patch 유사도 메모리. ([ISCA Archive][1]) 
* **B(Approach B)**: 도메인 시프트(현장/마이크/조건 변화)를 “메모리뱅크 + MemMixup + Domain Normalization”으로 완화. 타깃 정상 적어도 됨(3~10개) + 소스 정상 누적이 있으면 강력. ([ar5iv][2]) 
* **C(Approach C)**: 메타데이터(속도/노이즈/마이크위치 등)가 잘 들어오고, 소스 정상 데이터가 꽤 누적되면 “빠른 적응(3-shot)”을 학습으로 구현. 단, 초기엔 구현/운영 복잡도가 높음. ([ar5iv][3]) 
* “**95% 성능 자동 개발**”은 **조건부 목표**로 보는 게 합리적입니다. DCASE 벤치마크도 **소스 정상 ~1000개, 타깃 정상 3~10개** 같은 세팅을 쓰고(AUC/pAUC), 실제 현장에서는 **운전모드 커버리지/노이즈/마이크 변화**가 성패를 좌우합니다. ([ar5iv][4])

---

# 1) 상사가 말한 [ALP] 요구를 “데이터 Form + 자동 파이프라인”으로 번역

상사 요구를 그대로 “실행 가능한 프로세스”로 바꾸면 아래 5단계입니다.

## 1) 여러 소리를 커버하는 궁극 프로세스

* 여기서 “여러 소리” = **machine_type(소리 종류)**로 통일합니다.
* toy에서는 `fan, bearing, valve` 3개를 쓰지만, 실제 ALP는 6~7개로 확장합니다. (코드 구조상 동일 방식으로 확장 가능)

## 2) 6~7개 소리 선택

* 예: 팬/펌프/밸브/베어링/모터/컴프레서/기어박스 등.
* 선택의 핵심은 “**서로 소리 분포가 다른 그룹**”으로 묶어야 한다는 점입니다(각 그룹마다 정상 분포가 다름).

## 3) 고객에게 “Form” 제공 (문서가 아니라 **수량/조건 명세**)

이 Form은 각 소리 종류마다 고객이 준비해야 할 데이터 최소·권장 패키지를 정의합니다.

### Form이 반드시 포함해야 할 것 (요약)

* **타깃 정상음(필수)**: “그 현장, 그 마이크 위치, 그 운전조건”에서의 정상 기계음
* **현장 노이즈-only(강력 권장)**: 기계가 꺼져있거나(가능하면), 또는 기계음이 최소인 상태의 배경소음
* **메타데이터(가능하면 필수)**: 자산/구간(section), 마이크 위치, 운전조건(속도/부하), 날짜/시간대 등

DCASE 최신 설정도 “타깃 정상 10개 + 보조(클린/노이즈-only) 100개”처럼 **노이즈-only/클린 보조데이터를 별도 자원으로 취급**합니다. ([ar5iv][4])

## 4) 솔루션(딥러닝 프로세스)에 넣고 학습/적응

* ALP 관점에서 자동화하려면, **데이터 패키지(양/품질)에 따라 A/B/C 중 무엇을 돌릴지 자동 선택**해야 합니다.
* 즉, “Form”에서 받은 **정상 클립 수/노이즈 분량/메타데이터 품질**이 곧 라우팅 조건입니다.

## 5) “95% 성능 모델 자동 개발”

* 현실적으로 “95%”는 **정의가 필요**합니다. DCASE는 AUC와 pAUC(낮은 FPR 구간)를 공식적으로 씁니다. ([ar5iv][4])
* 따라서 ALP 내부 KPI는 예를 들어:

  * **AUC ≥ 0.95** (순위 성능)
  * **pAUC(FPR≤0.1) ≥ 0.90** (오탐 억제)
* 다만, 이 수준은 보통 **충분한 정상 분포 커버(시간 단위)**가 있을 때 더 현실적입니다. 예를 들어 MIMII는 모델당 정상음이 **5000~10000초(약 1.4~2.8시간)** 수준으로 구성됩니다. ([Zenodo][5])
* 즉 “자동으로 95%”는 **데이터 요구량/조건이 충족될 때** 가능한 목표로 두는 게 합리적입니다.

---

# 2) 이번 4개 파일이 ALP에서 맡는 역할 (Form과 직접 연결)

## (1) toy_dummy_data.py = “고객이 주는 데이터가 이런 형태면 좋다”를 코드로 강제하는 샘플 생성기

이 파일은 “현장 문제를 흉내낸 가짜 데이터”를 만듭니다.

* 기본 표준: **16kHz / 10초 클립**을 디폴트로 둡니다. 
* 소리 종류: `machine_types` (toy에서는 fan/bearing/valve). 
* 설비/구간 단위: `sections_per_machine`(section_id)로 개별 개체를 흉내냅니다. 
* 도메인 시프트를 “노브”로 갖고 있습니다:

  * 속도 레벨, 부하 레벨, 노이즈 종류, 마이크 위치, SNR 범위 등 
  * 실제 생성에서도 source/target에 따라 speed/load/noise/mic이 다르게 샘플링됩니다. 

### “Form의 수량”과 직결되는 부분: DCASE 2021/2025를 모사한 카운트 스펙

* 이 toy 생성기는 기본 카운트를 **DCASE 2025 first-shot 스타일(소스 정상 990, 타깃 정상 10, 보조 100, 테스트 각 50/50 등)**로 정의해 두고, `--scale`로 축소합니다. 
* 특히 `scaled_counts()`에서 **타깃 정상 최소 3개**를 강제(=DCASE 2021 타깃 3-shot을 반영)합니다. 
* DCASE 2025는 실제로 학습에 **source normal 990 + target normal 10**을 쓰고, **supplementary 100(클린 머신음 또는 노이즈-only)**를 제공한다고 명시합니다. ([ar5iv][4])
* DCASE 2021은 타깃 도메인에 **정상 3개 클립만 제공**하는 few-shot 조건을 명시합니다. ([ar5iv][4])

즉, toy_dummy_data는 “ALP가 고객에게 요구할 Form”을 **벤치마크 현실감으로 강제하는 장치**입니다.

### 노이즈-only가 “별도 자원”인 구조도 반영

* toy는 supplemental split에서 **noise-only 성격의 클립을 별도로 생성**합니다(`supp_type="noise"`). 
* 이 설계는 DCASE 2025의 “supplementary = clean or noise-only” 철학을 그대로 흉내냅니다. ([ar5iv][4])

---

## (2) approach_a.py = “정상 3~10개만으로 즉시 동작” (학습 없는) 엔진

A는 “Form에서 타깃 정상음이 아주 적게 들어오는 상황”을 정면으로 지원합니다.

### 핵심 아이디어 (논문 흐름)

* 사전학습 AST를 **freeze**하고,
* support(정상)에서 patch 임베딩을 저장해 **key-memory**를 만든 다음,
* query의 patch가 support key와 얼마나 유사한지(코사인)로 이상도를 계산합니다. 
  이 계열은 training-free few-shot ASD 흐름과 맞닿아 있습니다. ([ISCA Archive][1])

### 실제 코드에서 “Form”과 매칭되는 입력

* `support_domain`(보통 target)에서 정상 클립 N개를 뽑아 support set 구성
* 이 때 N은 기본적으로 3~10이 현실적 (toy run도 섹션당 support=3)

### 특징 저장 방식(“무엇을 메모리에 저장하나?”)

* AST에서 **레이어별 patch token**을 가져옵니다(기본 layers=1..11). 
* support set의 모든 patch embedding을 이어붙여 **layer별 key bank**를 만듭니다. (이게 메모리)
* 장점: 학습이 없고 빠름. 단점: patch key가 커질 수 있어 `max_keys_per_layer` 같은 다운샘플링 옵션이 필요(코드에 포함). 

### 거리/유사도 방식

* query patch vs key bank를 **cosine similarity**로 계산하고,
* patch별로 가장 유사한 key를 찾은 뒤 `anomaly = 1 - max_cosine` 형태로 anomaly map을 만듭니다. 
* 대규모 key bank를 대비해 chunking으로 안정적으로 계산합니다. 

### 클립 점수 집계(“patch anomaly map을 한 점수로?”)

* `decision_quantile`을 tail fraction으로 두고, 상위 꼬리 평균으로 집계합니다. 예: q=0.05면 상위 5% 평균(가장 이상한 패치들만 반영). 
* 이게 현장 직관과 잘 맞습니다: “짧은 순간의 이상”이 전체 평균에 묻히지 않게.

---

## (3) approach_b.py = “현장/도메인 시프트가 상수”일 때의 메모리뱅크 엔진

B는 GenRep(ICASSP 2025 계열)에서 말하는 흐름과 유사한 구조를 toy로 구현한 것입니다.

### 핵심 아이디어 (논문 흐름)

* 사전학습 표현(논문에선 BEATs 같은 generic representation)을 쓰고, ([Proceedings of Machine Learning Research][6])
* 소스/타깃 정상 임베딩을 각각 memory bank로 저장한 뒤,
* 타깃이 적으면 MemMixup으로 늘리고, ([ar5iv][2])
* 점수 분포를 Z-score로 정규화해 도메인 간 스케일 차이를 맞춥니다. ([ar5iv][2])

### 이 코드에서 “무엇을 임베딩으로 쓰나?”

* toy 버전은 의존성을 줄이기 위해 기본 encoder를 `logmel_stats`로 둡니다. 
* 대신 옵션으로 `beats_speechbrain` 같은 외부 encoder를 붙일 수 있게 CLI 선택지를 열어둡니다. 
  (실서비스 성능을 노리면 BEATs 계열 같은 강한 프리트레인 표현을 고려하는 게 보통 유리합니다. ([Proceedings of Machine Learning Research][6]))

### 메모리뱅크 구현

* 소스 정상 임베딩 집합 `M_s`, 타깃 정상 임베딩 집합 `M_t`를 각각 저장.
* 테스트는 kNN 거리로 “정상 bank에서 얼마나 멀리 떨어졌는지”를 점수로 사용합니다.

### MemMixup(타깃 bank 증강) 구현 방식

* 타깃 임베딩이 너무 적을 때, 타깃 t와 소스 kNN을 섞어 가짜 타깃 샘플을 추가합니다. 
* 논문도 “타깃을 소스의 가까운 샘플과 보간”하는 형태로 서술하며 λ=0.9를 사용합니다. ([ar5iv][2])

### Domain Normalization(도메인 정규화)

* 소스 거리/타깃 거리를 각각 Z-score로 정규화하고,
* 최종 점수를 **min(z_s, z_t)**로 잡아 “어느 도메인에 더 가깝든, 가까운 쪽 기준으로 이상 여부 판단”을 구현합니다. 
* 학습 정상에서의 평균/표준편차를 추정할 때, 구현은 LOOCV(leave-one-out) 옵션도 갖고 있습니다(편향 완화 목적). 
* 논문도 도메인별 점수를 정규화해 분포를 맞추는 것을 핵심으로 둡니다. ([ar5iv][2])

---

## (4) approach_c.py = “메타데이터가 강할 때” few-shot 적응을 학습으로 밀어붙이는 엔진

C는 “고객이 메타데이터를 잘 줄 수 있는가?”가 승부처입니다.

### 이 접근이 요구하는 메타데이터(“Form”에 반드시 넣어야 하는 항목)

코드에서 메타 태스크로 사용하는 항목은 기본적으로:

* `section_id`
* `speed_level`
* `noise_id`
* `mic_position_id` 

또한 label을 뽑는 로직은 meta 컬럼에 직접 있거나, 없으면 `operating_condition` JSON에서 찾도록 구현되어 있습니다. 

즉, 고객 Form에는 “운전조건/마이크 위치/노이즈 조건”이 **반드시 구조화된 값으로** 들어와야 합니다.

### “어떻게 적응하나?” (핵심 알고리즘 직관)

* 이 논문 흐름은 “메타데이터 기반 보조 분류(auxiliary classification)” 과제를 만들어, 도메인 시프트에 빨리 적응하도록 meta-learning을 합니다. ([ar5iv][3])
* 실제로 해당 연구는 few-shot(예: 3-shot) 적응을 명시적으로 다룹니다. ([ar5iv][3])

### 코드에서의 학습/적응 방식(요약)

* **학습**: Reptile 기반 outer-loop meta-update로 임베딩 네트워크를 학습합니다. (toy 로그에서도 outer loop가 반복됨)
* **에피소드 구성**: 각 에피소드에서 메타태스크(예: speed_level)를 택하고, n-way/k-shot 분류 문제를 구성.
* **프로토타입 기반 손실**: class prototype을 만들고, query를 prototype과의 거리에 기반해 분류하도록 학습합니다. 
* **적응**: 타깃 support(예: 3개 정상)로 프로토타입을 재추정하고, 테스트를 prototype 거리로 점수화합니다.

---

# 3) “Form(데이터 요구량)”을 A/B/C 라우팅 관점으로 정리 (가장 실무적인 형태)

아래가 ALP에서 고객에게 내밀 “수량 중심 Form”의 핵심입니다. (문서 양식이 아니라 **필요량 스펙**)

## 공통 기본 단위 (권장)

* 10초 단위 클립(또는 원본에서 10초로 자동 슬라이스)
* mono(또는 자동 mono), 16kHz 표준화(파이프라인에서 처리)
  toy도 이 기본 단위를 강제합니다. 

## Form 항목 1: “타깃 정상음” (필수)

* 의미: **실제 운영 현장/마이크/조건**에서의 정상 기계음
* 최소(즉시 MVP, A 가능): **3클립(=30초)**

  * 근거: DCASE 2021에서 타깃 정상 학습이 3클립인 설정이 공식적으로 존재 ([ar5iv][4])
  * toy 코드도 타깃 최소 3을 강제 
* 권장(안정화, A/B에 유리): **10클립(=100초)**

  * 근거: DCASE 2025 first-shot이 타깃 정상 10으로 설계 ([ar5iv][4])

## Form 항목 2: “현장 노이즈-only” (강력 권장)

* 의미: **기계음이 없거나 최소인 상태에서의 배경소음**(공장 소음/바람/다른 설비 소리 등)
* 최소: **30~60초**
* 권장: **2~5분(마이크 위치마다)**
* 근거(방향성): DCASE 2025는 supplementary를 “클린 머신음 또는 노이즈-only”로 별도 제공해, 노이즈-only를 명시적 자원으로 취급 ([ar5iv][4])
* toy도 noise-only를 supplemental로 따로 만듭니다. 

## Form 항목 3: “소스 정상음(누적 DB)” (B/C에서 중요)

* 의미: 과거 현장/다른 라인/랩 데이터 등 “정상”이지만 타깃과 조건이 다른 데이터
* 최소(B를 굴릴 수 있는 수준): **수십~수백 클립**

  * toy에서는 scale을 적용해 section당 source train을 50개 수준으로 줄여도 돌아가게 했습니다(실행 로그와 일치). 
* 권장(성능 목표를 현실화): **~1000 클립(약 2.8시간)**

  * DCASE 2021은 소스 정상 학습을 “약 1000개”로 둡니다. ([ar5iv][4])
  * MIMII도 모델당 정상음이 5000~10000초(1.4~2.8시간)로 구성됩니다. ([Zenodo][5])

## Form 항목 4: “메타데이터” (C의 성패를 결정)

* 최소(A/B): asset/section, 마이크 위치 정도만 있어도 운영 가능
* C를 진짜로 쓰려면(권장):

  * `speed_level`(속도 구간)
  * `load_level`(부하 구간)
  * `mic_position_id`
  * `noise_id`(현장/시간대/라인 소음 구분)
  * `section_id`(설비/제품/라인 단위)
* 코드상 C는 이 태스크들을 실제로 학습과제로 사용합니다. 

---

# 4) “자동(!) 개발”을 과장 없이 현실화하려면: 라우팅/게이트(품질관리) 설계가 핵심

DCASE는 공식 지표로 **AUC, pAUC(FPR≤0.1)**를 쓰고, 공식 점수는 이들을 조합합니다. ([ar5iv][4])
따라서 ALP에서 “자동 개발”을 주장하려면 아래처럼 **데이터량→알고리즘 선택→리포트/KPI 게이트**가 자동이어야 합니다.

### ALP 자동 라우팅(권장)

1. **타깃 정상 < 10개**: 무조건 A부터 (학습 없는 방식이 과적합 리스크 최소) 
2. **소스 정상 DB가 누적 + 도메인 시프트가 자주 문제**: B 활성화(메모리뱅크+정규화) 
3. **메타데이터 품질이 좋고(속도/마이크/노이즈 등), 소스 정상도 충분**: C를 “고급 옵션”으로 적용 ([ar5iv][3])

### “95%”를 KPI로 쓰려면 (현실적 해석)

* 제안 KPI(예시): AUC≥0.95 & pAUC≥0.90
* 단, **정답 라벨(최소한 테스트 구간의 정상/이상 확인)** 없이는 이 KPI를 증명할 수 없습니다.
* 따라서 고객 Form에는 가능하면 “검증용 소량 라벨”도 포함시키는 게 현실적입니다:

  * 예: “현장 운영 중 확실히 정상이라고 확인된 구간 20개, 확실히 이상 의심 5개(확정이 아니어도 됨)”

---

# 5) DB(데이터) 규모를 “초/클립/스토리지”로 감 잡기

## (A) 벤치마크가 암묵적으로 가정하는 “정상 DB 크기”

* DCASE 2021: 소스 정상 약 1000 클립(10초라면 약 2.8시간), 타깃 정상 3클립 ([ar5iv][4])
* DCASE 2025: 소스 정상 990, 타깃 정상 10, supplementary 100 ([ar5iv][4])
* MIMII: 모델당 정상 5000~10000초(1.4~2.8h), 이상 약 1000초 ([Zenodo][5])

즉, “95%급”을 목표로 하는 순간 **정상 데이터는 결국 ‘시간 단위’**로 늘어나는 쪽이 자연스럽습니다.

## (B) 저장 용량(대략)

* 16kHz, mono, 16-bit PCM에서

  * 1초 ≈ 16,000 samples × 2 bytes = 32,000 bytes ≈ 31.25KB
  * 10초 ≈ 312.5KB ≈ 0.3MB
* 1000클립(10초) ≈ 300MB(대략)
* 7개 소리 × (각 소리당 section 10개) × 1000클립이면

  * 7×10×300MB = 21,000MB ≈ 21GB 수준(대략)
    (원본이 48kHz면 약 3배)

---

# 6) 결론: 이 코드들이 ALP 요구에 대해 “지금 당장 무엇을 증명했나 / 무엇이 아직 부족한가”

## 지금까지 증명한 것(의의)

* **데이터 Form(수량·조건) → 표준화 → A/B/C 실행 → 점수/지표 산출**까지 toy로 end-to-end로 연결됨.
* 특히 toy 데이터 생성기가 DCASE 2021/2025의 few-shot/first-shot 철학(타깃 3~10, 보조 noise-only)을 모사하도록 설계되어, “고객 Form을 벤치마크 현실감으로 설계”하는 데 근거가 생김.  ([ar5iv][4])

## 아직 부족한 것(솔직히)

* “95% 자동”을 말하려면:

  1. **실제 현장 데이터 기준의 KPI 게이트**가 필요(라벨/검증 설계 포함)
  2. B에서 **강한 프리트레인 표현(BEATs 등)**을 실제로 붙이고 운영 최적화(캐싱/벡터DB)해야 함 ([Proceedings of Machine Learning Research][6])
  3. 노이즈-only를 **정량적으로 어떻게 보정에 쓰는지**(threshold calibration, score normalization, noise-aware embedding) 고도화가 필요
  4. C는 메타데이터 품질/카디널리티 설계가 핵심이라, “고객이 메타를 얼마나 안정적으로 주는지”에 따라 적용 여부를 자동 판단해야 함 



[1]: https://www.isca-archive.org/interspeech_2025/wu25b_interspeech.html "ISCA Archive - Towards Few-Shot Training-Free Anomaly Sound Detection"
[2]: https://ar5iv.org/pdf/2409.05035 "[2409.05035] Deep Generic Representations for Domain-Generalized Anomalous Sound Detection"
[3]: https://ar5iv.labs.arxiv.org/html/2204.01905 "[2204.01905] Learning to Adapt to Domain Shifts with Few-shot Samples in Anomalous Sound Detection"
[4]: https://ar5iv.org/pdf/2506.10097 "[2506.10097] DESCRIPTION AND DISCUSSION ON DCASE 2025 CHALLENGE TASK 2: FIRST-SHOT UNSUPERVISED ANOMALOUS SOUND DETECTION FOR MACHINE CONDITION MONITORING"
[5]: https://zenodo.org/records/3384388?utm_source=chatgpt.com "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection"
[6]: https://proceedings.mlr.press/v202/chen23ag?utm_source=chatgpt.com "BEATs: Audio Pre-Training with Acoustic Tokenizers"
