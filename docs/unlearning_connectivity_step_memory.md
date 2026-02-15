# Unlearning Connectivity Step Memory

운영 원칙:
- Step 2/3/5는 MIA 없이 connectivity proxy만 측정
- Barrier가 충분히 낮은 설정(top-1~2)만 Step 6(MIA)로 전달
- 목적: MIA 비용 절감 + 가능성 높은 조건 정밀평가

## Step 1: Seed 모델 생성
- 동일 dense init checkpoint에서 시작
- 동일 Df/Dr 사용
- seed만 다르게 unlearning 수행하여 endpoint 2개 이상 확보
- 학습 목적함수(현재 구현): `L = retain_weight * L_Dr - forget_alpha * L_Df`
  - `Dr`는 gradient descent
  - `Df`는 gradient ascent 효과

## Step 2: Linear Interpolation Connectivity (MIA 없음)
- 경로: `theta(t) = (1-t) * theta1 + t * theta2`
- 평가:
  - retain(Dr) loss curve
  - retain accuracy curve
  - barrier: `max_t Lr(theta(t)) - max(Lr(theta1), Lr(theta2))`
  - 각 `t`에서 BN running stats를 Dr 배치로 재계산

## Step 3: Subspace Restriction + Linear (MIA 없음)
- 목적: connectivity가 특정 subspace에서만 성립하는지 진단
- mask 예시:
  - `|theta1 - theta2|` top-k%
  - retain saliency top-k%
- 평가: Step 2와 동일

## Step 4: MCU (Bezier control point) [미구현]
- 현재 코드 범위 밖

## Step 5: MCU + Mask [미구현]
- 현재 코드 범위 밖

## Step 6: Barrier 후보만 MIA 정밀평가 [미구현]
- endpoint + 소수의 t 샘플에서 MIA 측정

## Barrier Gate (실험 운영용)
- retain acc drop <= 1~2%p 또는
- retain loss barrier <= epsilon
