## TL;DR (보고용 한 장 요약)

* **toy_dummy_data.py**: “현장 데이터가 적고(특히 이상음 희소), 도메인 시프트(마이크/소음/운전조건 변화)가 있는” 상황을 **가짜로 현실감 있게 흉내 낸 오디오 데이터셋 생성기**입니다. DCASE Task2 계열처럼 **source/target 도메인 + 정상-only 학습 + target few-shot(최소 3개) + 테스트 정상/이상 + 보조(supplemental: noise-only/clean-machine)** 구조를 강제로 만들어줍니다. 
* **approach_a.py**: **학습(파인튜닝) 없이**, 사전학습 **AST(Transformer)**를 **freeze**하고 정상 few-shot을 “패치(조각) 특징 메모리”로 저장한 뒤, 테스트 소리가 정상과 **얼마나 덜 비슷한지(1 − max cosine similarity)**로 이상 점수를 만드는 **Training-free few-shot 방식**입니다. 
* **approach_b.py**: **메모리뱅크(kNN) 기반**으로 정상 임베딩을 쌓고 거리로 이상을 판단합니다. 타깃 정상 데이터가 너무 적으면 **MemMixup(타깃 임베딩을 소스 근처로 보간)**으로 타깃 bank를 늘리고, **Domain Normalization(z-score)**로 source/target 점수 스케일 차이를 보정합니다(GenRep 스타일). 
* **approach_c.py**: “메타데이터(속도/소음/마이크/섹션 등)”를 **보조 분류 과제(auxiliary task)**로 두고, few-shot 상황에 맞춘 **에피소드 학습(ProtoNet)** + (옵션) **Reptile 메타러닝**으로 “새 조건에 빨리 적응 가능한 임베딩”을 학습합니다. 이후 타깃 few-shot 정상으로 적응해서 이상 점수를 계산합니다. 

---

# 0) 이 4개 파일이 함께 하는 “전체 흐름” 

1. **우리가 원하는 상황을 흉내 낸 데이터셋을 만든다**

   * `toy_dummy_data.py` 실행 → 폴더 구조와 WAV, 메타데이터 CSV가 생성됩니다. 

2. **같은 데이터셋을 3가지 알고리즘(A/B/C)로 돌려본다**

   * A/B/C는 모두 `toy_dummy_data.py`가 만든 **metadata.csv(단일 진실원천)** 기반으로 “훈련/테스트/도메인/정상·이상”을 필터링해서 동작합니다. 

3. **각 접근은 결과를 scores/metrics로 저장**

   * 실행 로그에서도 `scores.csv`, `metrics.json`을 출력하는 구조가 확인됩니다. 

---

# 1) toy_dummy_data.py — “무슨 가짜 데이터를 만들었나?” (쉽게 + 핵심 디테일)

## 1.1 왜 존재하나 (한 줄)

A/B/C가 공통으로 요구하는 **데이터 계약(data contract)**을 강제로 맞춰주는 “현장 대체용 샘플 데이터 생성기”입니다. 즉, 진짜 현장 데이터가 없어도 파이프라인이 끝까지 굴러가게 합니다. 

## 1.2 만들어지는 데이터의 “현장스러운 구조”

이 파일은 DCASE Task2 계열을 흉내 내서 다음을 모두 포함합니다. 

* **machine_type**: 예) fan / bearing / valve (소리 종류) 
* **section_id**: 같은 소리 종류 안에서도 “개별 설비(개체/라인)”를 구분하는 ID (자산/개체 단위) 
* **domain**: source / target

  * 소스 도메인 = 기존 익숙한 조건
  * 타깃 도메인 = 마이크/소음/운전조건이 바뀐 새로운 조건
    → 여기서 “도메인 시프트”가 발생합니다. 
* **split**: train / test / supplemental 
* **라벨 정책**:

  * train은 **정상-only**(is_anomaly=False)로만 생성 (실제 문제 조건 반영) 
  * test는 normal/anomaly가 섞여 생성 (평가용) 
* **supplemental(보조 데이터)**:

  * “noise-only”(소음만) 또는 “clean machine”(거의 클린한 기계음)을 별도로 생성합니다. 

## 1.3 “도메인 시프트”를 무엇으로 구현했나 (핵심)

도메인 시프트를 “말”로만 하는 게 아니라, 실제로 오디오 생성 파라미터를 바꿉니다.

* **소음 종류 자체가 source/target에서 다름**

  * noise_types: white/pink/brown/hum/impulse/babble
  * source_noise_ids와 target_noise_ids가 다르게 설정되어 있습니다. 
* **SNR(소음 섞이는 정도)도 source/target가 다름**

  * source는 더 깨끗한 범위(예: 10~30dB), target은 더 어려운 범위(예: 0~20dB). 
* **마이크 위치도 source/target가 다름**

  * source_mic_positions vs target_mic_positions가 분리되어 있습니다. 
* **속도/부하(operating condition)도 도메인 별로 다르게 뽑아** synthetic 음원을 생성합니다(로그/메타에 speed_level, load_level로 저장). 

> 상사에게 직관적으로 설명하면:
> “같은 팬(fan)이라도 **target에서는** 마이크를 다른 위치에 달고, 소음 종류도 바꾸고, 소음이 더 크게 섞이게 해서 ‘현장이 바뀐 상황’을 흉내 낸 데이터입니다.”

## 1.4 “few-shot” 조건을 어떻게 구현했나 (숫자가 중요한 부분)

기본 카운트는 DCASE2025의 형태(소스 990, 타깃 10 등)를 따라가되, toy라서 `--scale`로 축소합니다. 

특히 **타깃 정상 학습 데이터는 최소 3개로 강제**합니다. (DCASE2021 few-shot=3을 의식한 설계) 

즉, 지금 로그에서 나온 것처럼 `--scale 0.05`로 만들면 (라운딩/최소값 반영으로)

* section당 source train normal ≈ 50개
* section당 target train normal = 3개(최소값)
* section당 test = (도메인 2개 × (정상+이상 각 5개)) = 20개
  가 됩니다. (실제로 approach 실행 로그도 support=3, test=20 / src_train=50, tgt_train=3로 찍힘) 

## 1.5 “이상음(고장)”을 어떻게 가짜로 넣었나

테스트 anomaly 클립에는 다음 유형 중 하나를 주입합니다. 

* **impulse_burst**: 짧은 충격성 노이즈(50~300ms)를 1~2번 터뜨림 
* **band_tone**: 특정 대역에 길게 톤(휘파람 같은 성분)을 올림(0.5~2s) 
* **bearing_rattle**: 충격+공진 커널로 “덜컹거림” 비슷한 패턴을 합성 
* **dropout**: 신호를 특정 구간에서 급격히 깎아(attenuation) 끊김/약화처럼 만듦 

그리고 이 주입된 구간은 **anomaly_regions(시간 구간)**로도 메타데이터에 저장됩니다(“어느 구간이 이상인지”를 toy에서라도 추적 가능). 

## 1.6 메타데이터는 어떤 열들을 갖나 (A/B/C가 공통으로 읽는 핵심)

`ClipMeta`가 이 프로젝트의 데이터 계약입니다. 
핵심 필드만 상사에게 요약하면:

* **clip_id**: 파일을 식별하는 ID
* **relative_path**: wav 파일 경로
* **machine_type / section_id**: 소리 종류 / 설비 개체
* **split(train/test/supplemental), domain(source/target)**
* **label(normal/anomaly), is_anomaly**
* **speed_level/load_level/mic_position_id/noise_id/snr_db**: 도메인 시프트/운전조건 메타
* **supp_type**: supplemental이면 noise인지 machine인지
* **QA 필드(qa_rms, qa_peak, clipping/silence 비율)**: 데이터 품질 체크용 

---

# 2) approach_a.py — “A 방식(학습 없이 AST 유사도)”을 상사에게 설명하는 법

## 2.1 A의 목적(왜 만들었나)

* 고객이 “정상 소리”를 **진짜 몇 개(예: 3~10개)**만 줄 수 있을 때,
* **학습/파인튜닝 없이도 바로** 이상 여부를 점수로 내는 “콜드스타트 엔진”입니다. 

## 2.2 A의 핵심 직관 (비유)

* 정상 소리 몇 개를 “**정상 소리 사전(사전학습 AST가 만든 특징 조각 모음)**”으로 저장합니다.
* 새 소리를 들어보고, 그 소리의 “작은 조각(패치)”들이 정상 사전에 있는 조각들과 **얼마나 닮았는지**를 봅니다.
* **닮은 조각이 거의 없으면 이상**으로 판단합니다.

## 2.3 실제 알고리즘 Step-by-step (코드와 1:1로 연결되는 설명)

### Step A1) 정상 support set을 준비

* 타깃 도메인 train에서 정상 클립 N개를 뽑습니다(현재 toy에서는 최소 3개가 됨).
  로그에서도 section별 `support=3`으로 찍혔습니다. 

### Step A2) AST로 patch-level 특징을 뽑는다 (freeze)

* 사전학습 AST의 **patch representation**을 뽑습니다. 
* 논문 세팅을 따라 patch grid를 **12×101(총 1212개 패치)**로 가정하고, 여러 레이어를 평균합니다. 

### Step A3) 정상 패치들을 “키 메모리(key memory)”로 저장

* support set에서 나온 **모든 패치 벡터**를 모아 “정상 패치 메모리”를 만듭니다. (여기가 학습 대신 하는 핵심 저장 단계) 

### Step A4) 테스트(쿼리) 패치별로 “정상과의 최대 유사도”를 계산

* 쿼리 패치 벡터 q와, 정상 키 메모리 k들 사이의 **cosine similarity**를 계산하고, 그중 **최대값(max)**만 취합니다. 
* 구현은 큰 메모리 폭발을 막기 위해 key를 chunk로 나눠 matmul로 최대 유사도를 갱신합니다. 

### Step A5) 패치 anomaly 값으로 변환 → anomaly map → clip score

* 패치별 이상도: `a = 1 - max_cosine_similarity` 
* 레이어별 anomaly map을 평균
* 마지막으로 clip 단위 점수는 anomaly map의 **상위 quantile(상위 꼬리)**를 pooling해서 만듭니다. 

> 상사에게 “왜 quantile이냐”를 쉽게 말하면:
> “이상은 보통 10초 전체가 아니라 **짧은 구간**에만 튀는 경우가 많습니다. 평균을 내면 묻혀버리니, ‘가장 수상한 부분 상위 몇 %’를 본다는 뜻입니다.”

---

# 3) approach_b.py — “B 방식(메모리뱅크 + kNN + MemMixup + 도메인 정규화)”을 상사에게 설명하는 법

## 3.1 B의 목적(언제 A 대신 쓰나)

A는 “정상 몇 개로 바로 동작”이 강점이지만,
현장에서는 **도메인 시프트가 상수**이고(마이크/소음/환경이 현장마다 달라짐),
한 현장의 target 정상은 적어도 **과거 여러 현장/조건의 source 정상은 누적**되는 경우가 많습니다.

B는 그 상황에서 **누적된 source 정상(많음)**을 적극 활용해 target few-shot을 보강하려는 전략입니다. 

## 3.2 B의 핵심 직관 (비유)

* 정상 소리를 “지문(임베딩)”으로 만들고, 지문을 **보관함(memory bank)**에 쌓습니다.
* 새 소리의 지문이 정상 보관함에서 **얼마나 멀리 떨어져 있는지**를 봅니다.
* 타깃 정상 지문이 너무 적으면, “소스에서 비슷한 지문을 가져와 살짝 섞어” 타깃 보관함을 보강합니다(MemMixup).

## 3.3 실제 알고리즘 Step-by-step

### Step B1) 임베딩 추출(encoder)

* 논문 원형은 BEATs 같은 대형 사전학습 오디오 모델을 쓰지만,
  이 구현은 **기본값으로 의존성 가벼운 LogMelStatsEncoder(넘파이 기반 통계 특징)**을 제공합니다. 
* (옵션) torch + speechbrain 환경이면 BEATs encoder도 붙일 수 있게 설계되어 있습니다. 

### Step B2) 메모리뱅크 구성

* **source 정상 bank M_s**: source domain train normal들을 임베딩으로 변환해 저장
* **target 정상 bank M_t**: target domain train normal(아주 적음)을 임베딩으로 변환해 저장 

실행 로그에서도 section별로 `src_train=50 tgt_train=3`이 찍혀, toy 환경에서 bank 규모가 확인됩니다. 

### Step B3) (옵션) MemMixup으로 타깃 bank를 “부풀리기”

* 타깃 임베딩 하나하나에 대해, 소스 bank에서 가까운 K개를 찾고,
* `lambda=0.9`로 타깃을 주로 유지하면서 소스를 조금 섞은 합성 임베딩을 만들어 target bank에 추가합니다. 

> 상사에게 쉽게 말하면:
> “타깃 정상 샘플이 3개면 너무 빈약해서, 과거 데이터(소스) 중 비슷한 걸 섞어서 ‘타깃 정상의 주변 공간’을 조금 채워주는 겁니다.”

### Step B4) kNN 거리로 이상 점수

* 테스트 임베딩 y에 대해

  * 소스 bank까지의 kNN 평균 제곱거리 `d_s(y)`
  * 타깃 bank까지의 kNN 평균 제곱거리 `d_t(y)`
    를 계산합니다. 

### Step B5) (옵션) Domain Normalization(DN)로 스케일 보정 후 결합

* source/target은 분포가 달라서 거리값 스케일이 쉽게 달라집니다.
* 그래서 training normal에서 계산한 거리 통계를 이용해 각각 z-score 정규화하고,
  최종 점수는 **`min(z_s(y), z_t(y))`**처럼 더 “가까운 도메인 기준”을 취합니다. 

---

# 4) approach_c.py — “C 방식(메타데이터 기반 적응/메타러닝)”을 상사에게 설명하는 법

## 4.1 C의 목적(왜 B만으로는 부족할 때가 있나)

현장에서 어려운 케이스는 이런 경우입니다.

* 마이크 위치/속도/부하/소음이 바뀌면 정상 소리도 달라져서,
  단순 거리 기반(A/B)은 **오탐**이 늘 수 있음.
* 그런데 고객이 “이 클립은 속도 레벨 2, 마이크 3, 소음 5” 같은 **메타정보**를 꽤 잘 준다면,
  그 메타정보를 이용해 “조건 변화에도 흔들리지 않는 표현(임베딩)”을 학습할 수 있음.

C는 바로 그 전략입니다. 

## 4.2 C가 쓰는 “메타데이터”는 정확히 무엇인가

이 구현에서 예시로 든 보조 과제 라벨은 다음입니다:

* section ID
* speed level
* noise ID
* microphone position 

toy_dummy_data가 이 필드를 모두 메타에 저장해둡니다(ClipMeta에 있음). 

## 4.3 C의 전체 Step-by-step

### Step C1) 특징 추출: log-mel

* 입력 wav → log-mel 스펙트로그램(설정: sample_rate, n_mels, win/hop, n_fft 등) 

### Step C2) 작은 신경망(Conv encoder)로 임베딩을 만든다

* log-mel을 받아 **embedding_dim** 차원의 벡터로 만드는 Conv 기반 encoder를 구성합니다. 

### Step C3) “에피소드 학습(ProtoNet)”으로 few-shot 상황을 흉내 내며 학습

* 정상 소리만 가지고도, 메타데이터를 클래스 라벨처럼 취급해 “분류 문제”를 만들 수 있습니다.
* 예: speed_level이 0/1/2라면, “이 클립은 속도 1”을 맞히는 보조 분류 과제로 학습
* 이렇게 하면 임베딩이 “속도/마이크/소음 변화에 의미 있게 반응”하도록 정리됩니다. 

### Step C4) (옵션) Reptile 메타러닝으로 “빠른 적응”을 강화

* 여러 task를 번갈아가며 Reptile로 업데이트하면, 나중에 target few-shot에서도 더 빨리 맞추도록 유도할 수 있습니다. 

### Step C5) 타깃 few-shot 정상으로 “적응(adapt)” 후 이상 점수 계산

* 타깃 도메인의 정상 support 몇 개로 prototype을 만들거나(또는 짧게 finetune),
* 테스트가 prototype에서 멀어지면 이상(거리 기반 or nll 기반)으로 점수를 줍니다. 

---

# 5) “적용 시나리오”를 아주 쉽게 말하면 (A/B/C 선택 기준)

## 시나리오 1) 고객이 정말 데이터가 없고, 빨리 MVP가 필요

* 고객 제공: “정상 10초 클립 3~10개” + (가능하면) “현장 소음만 30~120초”
* 우리가 하는 일: **Approach A**로 바로 스코어링/알람 임계값만 잡아서 MVP
* 이유: 학습이 없어서 과적합 리스크가 낮고 즉시 동작 

## 시나리오 2) 고객/현장이 늘어나고, 과거 정상 데이터가 누적되기 시작

* 고객 제공: 새 현장 타깃 정상은 적지만, 이미 다른 현장/과거 조건의 정상은 많음
* 우리가 하는 일: **Approach B**로 소스 정상 bank + 타깃 정상 bank, 필요시 MemMixup + DN
* 이유: “누적 정상 데이터”를 적극 활용해 타깃 few-shot 빈약함을 완화 

## 시나리오 3) 고객이 메타데이터를 잘 준다(속도/부하/마이크/소음)

* 고객 제공: 정상 데이터 + 메타데이터 품질이 좋음(운전조건 레이블 등)
* 우리가 하는 일: **Approach C**로 조건 변화에 강한 임베딩을 학습하고 few-shot 적응
* 이유: 메타 정보가 있으면 “도메인 시프트 대응”을 학습적으로 다룰 수 있음 

---

# 6) 아주 냉정한 정리 (toy의 의의와 한계)

## 이 toy가 “유의미한 이유”

* **데이터 계약(메타데이터 중심) + 3개 알고리즘 엔진**이 “한 번에” 굴러가는 구조를 만들었습니다.
* 실제 현장 데이터가 들어오면, 이 toy 구조를 그대로 “실데이터 로더/DB”로 바꾸는 것이 가능해집니다(핵심은 `metadata.csv` 같은 단일 진실원천 설계). 

## 하지만 과장하면 안 되는 한계

* 지금 데이터는 **합성**이라, 실제 공장/현장 소리의 복잡함(비정상 패턴 다양성, 장비 고유음, 반사/잔향, 비정상 발생 확률 등)을 완전히 대변하지 못합니다.
* 따라서 **toy에서의 성능 수치가 실전 95%를 의미하지 않습니다.** (이건 “파이프라인이 돌아간다”는 기술 리스크 제거 단계에 가깝습니다.)

---
