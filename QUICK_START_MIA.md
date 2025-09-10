# 🚀 DWA MIA 평가 빠른 시작 가이드

현재 `runs/` 폴더에 있는 DWA 훈련 결과로 MIA 평가를 실행하는 방법입니다.

## 📁 현재 상황
```bash
runs/dwa/
├── kill_active_plain_dead/     # 5개 실험
├── kill_and_reactivate/        # 5개 실험  
└── reactivate_only/            # 5개 실험
```

✅ **발견된 실험**: 총 **15개의 완료된 DWA 실험** (각 모드별 sparsity 0.5, 0.7, 0.8, 0.9, 0.95)

⚠️ **주의**: 현재 모델 체크포인트 파일(`.pth`)이 없는 상태입니다. DWA 훈련시 모델 저장이 되지 않은 것 같습니다.

## 🎯 실행 방법

### 1. 단일 실험 MIA 평가

특정 조건의 실험 하나에 대해 MIA 평가:

```bash
# 기본 실행 (첫 번째 찾은 실험 사용)
python run_single_mia.py

# 특정 조건 지정
python run_single_mia.py --dataset cifar10 --model resnet18 --mode kill_active_plain_dead --sparsity 0.9

# GPU 지정
python run_single_mia.py --gpu 1 --mode kill_active_plain_dead --sparsity 0.8
```

### 2. 전체 배치 MIA 평가

모든 실험에 대해 자동으로 MIA 평가:

```bash
# 🎯 한번에 모든 실험 실행 (권장)
python run_all_mia.py

# 특정 조건으로 필터링
python run_all_mia.py --filter_mode kill_active_plain_dead
python run_all_mia.py --filter_sparsity 0.9

# 또는 배치 스크립트 직접 사용
python run_batch_mia.py --filter_mode kill_active_plain_dead

# 실험 목록만 확인 (실행하지 않음)
python run_batch_mia.py --dry_run
```

### 3. 고급 옵션

```bash
# 특정 공격만 실행
python run_batch_mia.py --attacks samia,threshold

# 변환 과정 생략 (이미 변환된 경우)
python run_batch_mia.py --skip_conversion

# 조합 사용
python run_batch_mia.py --filter_mode kill_active_plain_dead --filter_sparsity 0.9 --attacks threshold --gpu 1
```

## 📊 결과 확인

### 실행 후 생성되는 파일들:

1. **WeMeM 호환 구조**: `result/cifar10_resnet18/`
   - `victim_model/best.pth`
   - `shadow_model_*/best.pth` 
   - `l1unstructure_*_model/best.pth`
   - `data_prepare.pkl`

2. **MIA 결과 로그**: `log/cifar10_resnet18/`
   - `l1unstructure_0.9_.txt` (sparsity별)

3. **배치 결과**: `batch_mia_results_YYYYMMDD_HHMMSS.json`

### 결과 해석:
```
Victim pruned model test accuracy: 89.37%
SAMIA attack accuracy: 67.890%
Conf attack accuracy: 62.340%
Entr attack accuracy: 59.120%
...
```

- **> 60%**: 모델이 membership 정보 누출 (위험)
- **≈ 50%**: 랜덤 추측 수준 (안전)

## 🔧 문제 해결

### 1. 모델 체크포인트 없음
현재 상황입니다. 스크립트는 이를 자동으로 감지하고 처리하려고 시도합니다.

```bash
# 실험 발견 확인
python run_batch_mia.py --dry_run
```

### 2. GPU 메모리 부족
```bash
# 배치 크기 줄이기 (configs/*.json 파일에서)
# "batch_size": 64 또는 32
```

### 3. 의존성 문제
```bash
pip install torch torchvision numpy pickle-mixin
```

## ⚡ 빠른 테스트

현재 상태에서 바로 테스트:

```bash
# 1. 실험 목록 확인
python run_batch_mia.py --dry_run

# 2. 하나만 실행 테스트
python run_single_mia.py --mode kill_active_plain_dead --sparsity 0.9

# 3. 전체 실행 (주의: 시간 오래 걸림)
python run_all_mia.py

# 또는 특정 조건으로 필터링
python run_all_mia.py --filter_mode kill_active_plain_dead
```

## 📝 다음 단계

1. **모델 체크포인트 문제 해결**: DWA 훈련 코드에 모델 저장 로직 추가 필요
2. **결과 분석**: 생성된 MIA 결과 분석 및 시각화
3. **실험 확장**: 다양한 MIA 공격 기법 테스트

---

문제가 있으면 실행 로그를 확인하고, 각 단계별로 오류 메시지를 체크하세요!