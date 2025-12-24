# 항공권 동적 가격 결정 MDP (Markov Decision Process)

## 프로젝트 개요

항공권 판매에서 수익을 최대화하기 위한 동적 가격 결정 문제를 MDP로 모델링하고, 다양한 정책(Policy)을 비교 분석하는 프로젝트입니다.

### 핵심 모델

- **상태(State)**: `c_t` - 잔여 좌석 수
- **행동(Action)**: `p_t` - 설정 가격 (이산 유한 집합)
- **외생 정보**:
  - 고객 도착: `M_t ~ Poisson(μ_t)` (시간에 따라 증가하는 지수 수요 모델)
  - 구매 확률: 다항 로짓 모델 (자사/경쟁사/미구매 선택)
- **상태 전이**: `c_{t+1} = c_t - s_t` (판매량만큼 좌석 감소)
- **보상**: `r_t = p_t × s_t` (가격 × 판매량)

---

## 파일 구조

```
code_submission/
├── model.py    # MDP 환경 클래스 (AirlinePricingModel)
├── policy.py   # 정책 클래스들 (DP, Greedy, Rollout 등)
├── driver.py   # 시뮬레이션 드라이버 (메인 진입점)
├── plot.py     # 시각화 유틸리티
└── README.md   # 본 문서
```

---

## 의존성 설치

```bash
pip install numpy scipy matplotlib
```

---

## 실행 방법

### 기본 실행

```bash
# DP(Backward Induction) 정책으로 100회 시뮬레이션 (기본값)
python driver.py

# 도움말 보기
python driver.py --help
```

### 정책별 실행

```bash
# DP (Backward Induction) 정책
python driver.py -p dp

# Greedy (근시안적) 정책
python driver.py -p greedy

# Rollout (Monte Carlo 기반) 정책
python driver.py -p rollout

# Fixed Price (고정 가격) 정책
python driver.py -p fixed --fixed-price 15.0

# Random (무작위) 정책
python driver.py -p random
```

### 시나리오 선택

```bash
# 시나리오 1: 인천-뉴욕 노선 (비즈니스 수요)
python driver.py --scenario 1 -p dp

# 시나리오 2: 인천-호놀룰루 노선 (레저 수요)
python driver.py --scenario 2 -p dp
```

### 실행 모드

```bash
# 단일 에피소드 실행
python driver.py --mode single -p greedy

# 다중 에피소드 실행 (통계 계산)
python driver.py --mode multiple -n 500 -p dp

# 둘 다 실행
python driver.py --mode both -p dp
```

---

## 주요 인자

| 인자 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--policy` | `-p` | 정책 선택 (fixed, random, greedy, dp, rollout) | dp |
| `--scenario` | - | 시나리오 선택 (1: 뉴욕, 2: 호놀룰루) | 1 |
| `--mode` | `-m` | 실행 모드 (single, multiple, both) | multiple |
| `--episodes` | `-n` | 시뮬레이션 횟수 | 100 |
| `--seed` | `-s` | 랜덤 시드 | 42 |
| `--fixed-price` | - | Fixed 정책 가격 설정 | 시나리오별 평균 |

---

## 정책 설명

### 1. DP (Backward Induction)
- Bellman 방정식 기반 **정확한 최적 정책**
- `V(t,c) = max_p { E[r + γ·V(t+1, c')] }` 계산
- 전체 상태 공간에 대해 최적 가격 테이블 사전 계산

### 2. Greedy (Myopic)
- **즉시 기대 수익만 최대화**하는 근시안적 정책
- 미래를 고려하지 않음
- 빠르지만 최적이 아님

### 3. Rollout
- Monte Carlo 시뮬레이션 기반 **근사 정책**
- 각 가격에 대해 N번 시뮬레이션 후 평균 수익이 가장 높은 가격 선택
- Base Policy(Greedy)를 사용하여 미래 스텝 시뮬레이션

### 4. Fixed Price
- 전 기간 **동일 가격** 유지
- 베이스라인 비교용

### 5. Random
- **무작위 가격** 선택
- 테스트용

---

## 시나리오 파라미터

### 시나리오 1: 인천-뉴욕 (비즈니스 수요)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 좌석 수 | 1200 | 총 판매 가능 좌석 |
| 판매 기간 | 90일 | 출발까지 남은 일수 |
| 가격 범위 | 0 ~ 30 | 단위: 십만원 |
| 가격 민감도(β) | 0.6 | 낮은 탄력성 (비즈니스) |
| 경쟁사 가격 | 8 ~ 18 | 시간에 따라 증가 |

### 시나리오 2: 인천-호놀룰루 (레저 수요)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 좌석 수 | 600 | 총 판매 가능 좌석 |
| 판매 기간 | 90일 | 출발까지 남은 일수 |
| 가격 범위 | 0 ~ 30 | 단위: 십만원 |
| 가격 민감도(β) | 1.2 | 높은 탄력성 (레저) |
| 경쟁사 가격 | 6 ~ 18 | 시간에 따라 증가 |

---

## 출력 결과

실행 시 `fig/` 디렉토리에 다음 그래프가 저장됩니다:

| 파일명 | 설명 |
|--------|------|
| `*_episode_result.png` | 단일 에피소드 결과 (좌석, 가격, 판매량, 수익) |
| `simulation_stats.png` | 다중 에피소드 통계 (평균 좌석, 가격, 수익 분포) |
| `*_policy_3d.png` | DP 정책 3D 표면 그래프 |
| `*_policy_heatmap.png` | DP 정책 히트맵 |

---

## 코드 구조 상세

### model.py - AirlinePricingModel

```python
# 주요 메서드
model.step(c_t, p_t, rng, t)        # 한 스텝 실행
model.purchase_probability(p_t)     # 다항 로짓 구매 확률
model.sample_arrivals(rng, t)       # 포아송 도착 샘플링
model.get_competitor_price(t)       # 동적 경쟁사 가격
```

### policy.py - 정책 클래스

```python
# 공통 인터페이스
policy.select_action(state, t)  # 가격 선택
policy.reset()                  # 에피소드 초기화

# DP 정책 전용
dp_policy.solve()               # Backward Induction 실행
dp_policy.get_value(state, t)   # 가치 함수 조회
```

### driver.py - Simulator

```python
# 시뮬레이션 실행
simulator.run_episode()              # 단일 에피소드
simulator.run_multiple(n_episodes)   # 다중 에피소드 + 통계
```

---

## 예제 출력

```
==================================================
[시나리오1: 인천-뉴욕 (비즈니스)]
Model Parameters:
  Seats: 1200, Stages: 90
  Price Range: 0.0 ~ 30.0 (십만원)
  Beta (가격 민감도): 0.6
==================================================

[Policy: DP]
  Running Backward Induction...
  Solving t=80
  Solving t=70
  ...
  Backward Induction completed.
  Initial Value V(0, 1200): 15,234,567

[Multiple Episodes]
Episode 100/100
  Mean Revenue: 14,892,345
  Std Revenue: 234,567
  Min Revenue: 14,123,456
  Max Revenue: 15,567,890

Done!
```

---

## 수요 모델 수식

### 지수 수요 모델 (시간에 따른 도착)

```
cumulative(t) = A × exp(-β × (T - t))
daily_demand[t] = cumulative(t+1) - cumulative(t)
```

- `t=0`: 판매 시작 (수요 낮음)
- `t=T-1`: 출발 직전 (수요 높음)

### 다항 로짓 모델 (구매 확률)

```
P(구매) = exp(U_me) / (exp(U_me) + exp(U_comp) + exp(U_no_buy))

U_me = V_me - β × p_me       # 자사 효용
U_comp = V_comp - β × p_comp # 경쟁사 효용
U_no_buy = V_no_buy          # 미구매 효용
```

---

## 참고사항

- matplotlib은 `Agg` 백엔드 사용 (GUI 없이 파일 저장만)
- 난수 시드 고정으로 결과 재현 가능
- DP 정책은 상태 공간이 클수록 계산 시간 증가
