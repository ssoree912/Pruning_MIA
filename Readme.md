# Neural Network Pruning with Comprehensive MIA Vulnerability Analysis

통합 훈련 및 멤버십 추론 공격(MIA) 취약성 분석을 위한 신경망 가지치기 프로젝트입니다. Dense, Static, Dynamic Pruning with Feedback(DPF) 방법론을 비교하고, 각 방법의 MIA 취약성을 종합적으로 평가합니다.

## 🎯 주요 특징

### 지원하는 Pruning 방법
- **Dense**: 기본 밀도 모델 (baseline)
- **Static**: 정적 가지치기 (가중치 제거 후 고정)
- **DPF**: Dynamic Pruning with Feedback (동적 재활성화 지원)

### 지원 스파시티 레벨
- 50%, 70%, 80%, 90%, 95% 스파시티

### 종합적인 MIA 평가
**Advanced MIA Attacks**:
- LiRA (Likelihood Ratio Attack)
- Shokri-NN Attack
- Top3-NN Attack  
- Class Label-NN Attack
- SAMIA Attack

**WeMeM MIA Attacks**:
- Confidence-based Attack
- Entropy-based Attack
- Modified Entropy Attack
- Neural Network-based Attack

## 🚀 빠른 시작

### 전체 실험 실행 (훈련 + MIA 평가)
```bash
# 기본 설정으로 모든 방법 훈련 및 MIA 평가
python train.py

# 특정 방법만 훈련
python train.py --methods dense static --sparsities 0.7 0.8 0.9

# Wandb 로깅 활성화
python train.py --wandb --wandb-project my-project --wandb-entity my-username
```

### 개별 모델 훈련
```bash
# Dense 모델
python run_experiment.py --name dense_cifar10 --dataset cifar10 --epochs 200

# Static 가지치기 (80% 스파시티)
python run_experiment.py --name static_80 --prune --prune-method static --sparsity 0.8

# DPF 가지치기 (90% 스파시티)
python run_experiment.py --name dpf_90 --prune --prune-method dpf --sparsity 0.9
```

### MIA 평가만 실행
```bash
# 기존 훈련된 모델들에 대해 종합 MIA 평가
python mia/unified_mia_evaluation.py --runs-dir ./runs

# Advanced MIA만 실행
python mia/mia_advanced.py --runs-dir ./runs

# WeMeM MIA만 실행  
python mia/mia_wemem.py --runs-dir ./runs
```

### 테스트 실행
```bash
# MIA 평가 파이프라인 테스트
python test_mia.py --runs-dir ./runs
```

## 📁 프로젝트 구조

```
├── train.py                       # 통합 훈련 및 MIA 평가 스크립트
├── run_experiment.py              # 개별 실험 실행기
├── test_mia.py                    # MIA 평가 테스트 스크립트
│
├── configs/                       # 설정 관리
│   ├── config.py                 # 실험 설정 클래스
│   └── __init__.py
│
├── models/                        # 모델 정의
│   ├── resnet.py                 # ResNet 구현
│   ├── wideresnet.py             # Wide ResNet 구현
│   └── __init__.py
│
├── mia/                          # MIA 평가 모듈
│   ├── mia_advanced.py           # Advanced MIA 공격들
│   ├── mia_classic.py              # WeMeM MIA 공격들
│   ├── unified_mia_evaluation.py # 통합 MIA 평가
│   └── run_mia_evaluation.py     # 개별 모델 MIA 평가
│
├── runs/                         # 훈련된 모델 저장소
│   ├── dense/cifar10/           # Dense 모델 결과
│   ├── static/sparsity_X/cifar10/ # Static 가지치기 결과
│   └── dpf/sparsity_X/cifar10/  # DPF 가지치기 결과
│
├── results/                      # MIA 평가 결과
│   ├── mia/                     # 통합 MIA 결과
│   ├── advanced/                # Advanced MIA 결과
│   └── wemem/                   # WeMeM MIA 결과
│
├── training_results.csv          # 훈련 성능 요약
└── comprehensive_mia_results.csv # 종합 MIA 취약성 결과
```

## 🔬 방법론 비교

### Dense Model
- 기본 밀도 신경망
- 모든 가중치 활성화 상태
- 높은 정확도, 높은 MIA 취약성

### Static Pruning
```python
# Forward pass
output = input * mask

# Backward pass  
gradient = gradient * mask  # 제거된 가중치는 영원히 0
```
- 가지치기 후 마스크 고정
- 희소한 그래디언트 업데이트
- 메모리 효율적, 중간 수준의 정확도

### DPF (Dynamic Pruning with Feedback)
```python
# Forward pass
output = input * mask

# Backward pass
gradient = gradient  # 전체 그래디언트 계산 (재활성화 가능)
```
- 동적 가중치 재활성화
- 밀도 그래디언트 업데이트
- Static보다 높은 정확도, 복잡한 훈련 과정

## 📊 결과 분석

### 훈련 성능 결과
- `training_results.csv`: 각 방법별 정확도, 손실, 훈련 시간
- `runs/*/experiment_summary.json`: 상세 훈련 메트릭

### MIA 취약성 결과
- `comprehensive_mia_results.csv`: 모든 MIA 공격 결과 통합
- `results/mia/advanced/`: LiRA, Shokri-NN 등 고급 공격 결과
- `results/mia/wemem/`: Confidence, Entropy 기반 공격 결과

### 주요 메트릭
```csv
experiment,method,sparsity,lira_auc,confidence_accuracy,neural_network_auc,samia_accuracy,...
dense_cifar10,dense,0.0,0.85,0.78,0.82,0.76,...
static_sparsity_0.8_cifar10,static,0.8,0.72,0.65,0.68,0.63,...
dpf_sparsity_0.8_cifar10,dpf,0.8,0.74,0.67,0.71,0.65,...
```

## ⚙️ 설정 옵션

### 훈련 설정
```bash
python train.py \
    --methods dense static dpf \
    --sparsities 0.5 0.7 0.8 0.9 0.95 \
    --dataset cifar10 \
    --arch resnet \
    --epochs 200 \
    --skip-existing
```

### Wandb 로깅
```bash
python train.py \
    --wandb \
    --wandb-project dcil-pytorch \
    --wandb-entity your-username \
    --wandb-tags pruning mia-analysis cifar10
```

### MIA 평가 설정
```bash
python mia/unified_mia_evaluation.py \
    --runs-dir ./runs \
    --results-dir ./results/unified_mia
```

## 📈 예상 결과

### 모델 성능 (정확도)
```
Dense > DPF > Static (동일 스파시티에서)
낮은 스파시티 > 높은 스파시티
```

### MIA 취약성 (낮을수록 좋음)
```
Dense (가장 취약) > DPF ≈ Static (공격 유형에 따라 차이)
높은 스파시티에서 취약성 감소 경향
```

### Privacy-Utility Trade-off
- Dense: 높은 성능, 높은 취약성
- Static: 중간 성능, 중간 취약성  
- DPF: 높은 성능, Static과 유사한 취약성

## 🛠️ 의존성

### 필수 패키지
```bash
torch>=1.9.0
torchvision
numpy
pandas  
scikit-learn
matplotlib
seaborn
wandb  # 선택사항
```

### 설치
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn wandb
```

## 📋 사용 예시

### 1. 빠른 프로토타이핑
```bash
# Dense와 Static 80%만 비교
python train.py --methods dense static --sparsities 0.8 --epochs 50
```

### 2. 전체 벤치마크
```bash
# 모든 방법과 스파시티 레벨로 완전한 실험
python train.py --epochs 200 --wandb
```

### 3. 특정 MIA 공격만 평가
```bash
# LiRA 공격만 실행
python mia/mia_advanced.py --attacks lira --runs-dir ./runs
```

## 🔧 트러블슈팅

### 일반적인 문제들

1. **메모리 부족**
   - 배치 크기 줄이기: `--batch-size 64`
   - 더 작은 모델 사용: `--arch resnet --layers 20`

2. **Wandb 연결 문제**
   - 엔티티 확인: `--wandb-entity your-correct-username`
   - 프로젝트 권한 확인

3. **MIA 평가 실패**
   - 모델 파일 존재 확인: `ls runs/*/best_model.pth`
   - 설정 파일 확인: `ls runs/*/config.json`

### 로그 및 디버깅
```bash
# 상세 로그로 실행
python train.py --verbose

# 특정 실험만 재실행
python train.py --methods static --sparsities 0.8 --skip-existing
```

## 📚 참고 자료

- [LiRA Paper](https://arxiv.org/abs/2112.03570)
- [WeMeM Framework](https://github.com/example/wemem)
- [Neural Network Pruning Survey](https://arxiv.org/abs/2103.00249)
