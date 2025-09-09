# 🎯 DWA → WeMeM-main 스타일 MIA 완전 파이프라인

이 가이드는 **DWA 훈련 결과**를 **WeMeM-main 스타일 MIA 평가**와 연동하는 완전한 파이프라인입니다.

## 📋 **전체 파이프라인 개요**

```
DWA 훈련 결과 → 구조 변환 → WeMeM-main MIA 평가
runs/dwa/    → result/     → log/
```

## 🚀 **Step-by-Step 실행**

### **Step 1: DWA 모델 훈련** ✅ 
```bash
# DWA 모델들 훈련 (이미 완료되었다고 가정)
python train_dwa.py --dwa-modes reactivate_only kill_active_plain_dead kill_and_reactivate \
    --sparsities 0.5 0.8 0.9 --dataset cifar10 --epochs 50

# 결과 확인
ls runs/dwa/
```

### **Step 2: DWA → WeMeM-main 구조 변환** 🔄
```bash
# DWA 결과를 WeMeM-main 구조로 변환
python scripts/dwa_to_wemem_converter.py --runs_dir ./runs --dataset cifar10 --model resnet18

# 변환 결과 확인
ls result/cifar10_resnet18/
# 출력: victim_model/, shadow_model_0/, shadow_model_1/, ..., l1unstructure_0.6_model/, data_prepare.pkl
```

### **Step 3: WeMeM-main 스타일 MIA 평가** 🎯
```bash
# 원본 WeMeM-main mia_modi.py 사용
python mia_modi.py 0 configs/cifar10_resnet18.json \
    --dataset_name cifar10 --model_name resnet18 \
    --attacks samia,threshold,nn,nn_top3,nn_cls

# 결과 확인
cat log/cifar10_resnet18/l1unstructure_0.6_.txt
```

## 📁 **파일 구조 (변환 후)**

```
prunning/
├── runs/dwa/                           # DWA 원본 결과 (유지)
│   ├── reactivate_only/
│   ├── kill_active_plain_dead/
│   └── kill_and_reactivate/
├── result/                             # WeMeM-main 호환 구조 (새로 생성)
│   └── cifar10_resnet18/
│       ├── data_prepare.pkl            # MIA 데이터 분할
│       ├── victim_model/best.pth       # Victim 모델
│       ├── shadow_model_0/best.pth     # Shadow 모델들
│       ├── shadow_model_1/best.pth
│       ├── l1unstructure_0.6_model/    # "Pruned" 모델 (실제로는 DWA)
│       └── shadow_l1unstructure_0.6_model_0/
├── log/                                # MIA 결과 로그 (새로 생성)
│   └── cifar10_resnet18/
│       └── l1unstructure_0.6_.txt      # MIA 공격 결과
├── mia_modi.py                         # WeMeM-main MIA 평가 스크립트
├── attackers.py                        # MIA 공격 클래스
├── base_model.py                       # 베이스 모델 클래스
└── configs/cifar10_resnet18.json       # 설정 파일
```

## 🔧 **설정 파일 예시**

`configs/cifar10_resnet18.json`:
```json
{
  "device": 0,
  "config_path": "configs/cifar10_resnet18.json",
  "dataset_name": "cifar10",
  "model_name": "resnet18", 
  "num_cls": 10,
  "input_dim": 3,
  "batch_size": 128,
  "pruner_name": "l1unstructure",
  "prune_sparsity": 0.6,
  "shadow_num": 5,
  "attacks": "samia,threshold,nn,nn_top3,nn_cls",
  "original": false
}
```

## 📊 **MIA 공격 결과 해석**

실행 완료 후 `log/cifar10_resnet18/l1unstructure_0.6_.txt`에서 확인할 수 있는 결과:

```
Victim pruned model test accuracy: 85.234%
SAMIA attack accuracy: 67.890%
Conf attack accuracy: 62.340%
Entr attack accuracy: 59.120%
Mentr attack accuracy: 61.780%
Hconf attack accuracy: 64.560%
NN attack accuracy: 68.230%
Top3-NN Attack Accuracy: 65.670%
NNCl Attack Accuracy: 66.890%
```

### **해석 가이드**
- **> 60%**: 모델이 membership 정보를 누출 (위험)
- **≈ 50%**: 랜덤 추측 수준 (안전)
- **SAMIA**: 가장 정교한 공격, 보통 가장 높은 성공률
- **Threshold 공격**: 간단하지만 효과적
- **NN 공격**: 신경망 기반, 복합 정보 활용

## 🚨 **문제 해결**

### **1. Import 에러**
```bash
# 필요 패키지 설치
pip install torch torchvision numpy

# CIFAR 모델 문제시
pip install torchvision-models-cifar10
```

### **2. 구조 변환 실패**
```bash
# DWA 결과 확인
ls runs/dwa/*/sparsity_*/cifar10/

# 수동으로 변환 확인
python -c "from scripts.dwa_to_wemem_converter import convert_dwa_to_wemem_structure; convert_dwa_to_wemem_structure()"
```

### **3. GPU 메모리 부족**  
```bash
# 배치 크기 줄이기
# configs/cifar10_resnet18.json에서 "batch_size": 64로 변경
```

## 🎯 **완전 자동화 스크립트**

전체 파이프라인을 한 번에 실행:

```bash
#!/bin/bash
echo "🚀 DWA → WeMeM MIA Complete Pipeline"

# Step 1: 구조 변환
echo "🔄 Converting DWA to WeMeM structure..."
python scripts/dwa_to_wemem_converter.py --dataset cifar10 --model resnet18

# Step 2: MIA 평가
echo "🎯 Running MIA evaluation..."
python mia_modi.py 0 configs/cifar10_resnet18.json --attacks samia,threshold,nn,nn_top3,nn_cls

# Step 3: 결과 출력
echo "📊 Results:"
cat log/cifar10_resnet18/l1unstructure_0.6_.txt

echo "✅ Pipeline completed!"
```

## ✨ **추가 옵션**

### **여러 데이터셋 동시 평가**
```bash
for dataset in cifar10 cifar100; do
    python scripts/dwa_to_wemem_converter.py --dataset $dataset --model resnet18
    python mia_modi.py 0 configs/${dataset}_resnet18.json
done
```

### **다양한 공격 조합 테스트**
```bash
# Threshold 공격만
python mia_modi.py 0 configs/cifar10_resnet18.json --attacks threshold

# SAMIA + NN 공격
python mia_modi.py 0 configs/cifar10_resnet18.json --attacks samia,nn

# 모든 공격
python mia_modi.py 0 configs/cifar10_resnet18.json --attacks samia,threshold,nn,nn_top3,nn_cls
```

이제 DWA 훈련 결과를 완전히 WeMeM-main 스타일 MIA 평가와 연동할 수 있습니다! 🎉