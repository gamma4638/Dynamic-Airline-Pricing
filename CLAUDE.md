# Dynamic Airline Pricing

## 프로젝트 개요
항공권 동적 가격 결정 MDP (Markov Decision Process) 프로젝트.
포아송 도착 + 다항로짓 수요 모델 기반으로 최적 가격 정책을 계산한다.

## 실행 방법
```bash
# 기본 실행 (DP 정책, 1000회 시뮬레이션)
python code/driver.py

# 정책별 실행
python code/driver.py -p dp                        # DP (Backward Induction)
python code/driver.py -p greedy -n 500             # Greedy 정책, 500회
python code/driver.py -p rollout --scenario 2      # Rollout 정책, 시나리오2
python code/driver.py -p fixed --fixed-price 12    # Fixed Price 12(십만원)

# 모든 정책 비교 실험
python code/driver.py --compare

# 도움말
python code/driver.py --help
```

## 프로젝트 구조
```
code/
├── model.py     # MDP 환경 클래스 (AirlinePricingModel)
├── policy.py    # 모든 정책 클래스 (DP, Greedy, Rollout 등)
├── driver.py    # 시뮬레이션 드라이버 (Simulator 클래스, 메인 진입점)
└── plot.py      # 시각화 유틸리티

fig/             # 그래프 출력 디렉토리
reports/         # 분석 리포트
```

## 핵심 모델 (basic_model.md 참조)
- **State**: c_t (잔여 좌석 수)
- **Action**: p_t (가격, 이산 유한 집합)
- **Demand Model**: Q_t = M_t * P(buy_me), M_t ~ Poisson(mu)
- **Transition**: c_{t+1} = c_t - min(Q_t, c_t)
- **Reward**: r_t = p_t * min(Q_t, c_t)

## 의존성
- numpy
- matplotlib
- scipy (포아송 분포)

## 기본 파라미터 (driver.py)
| 파라미터 | 시나리오1 (뉴욕) | 시나리오2 (호놀룰루) | 설명 |
|---------|-----------------|-------------------|------|
| num_seats | 300 | 300 | 총 좌석 수 |
| num_stages | 90 | 90 | 판매 기간 |
| price_min/max | 10.0/20.0 | 6.0/10.0 | 가격 범위 (십만원) |
| mu | 20 | 20 | 포아송 평균 도착 |
| beta | 0.204 | 0.191 | 가격 민감도 |

## 코딩 컨벤션
- 한글 주석 사용 가능
- numpy 스타일 docstring
- 클래스명: PascalCase
- 함수/변수명: snake_case
- 수학 변수는 논문 표기 따름 (c_t, p_t, Q_t 등)

## 주의사항
- matplotlib은 `Agg` 백엔드 사용 (GUI 없이 저장)
- 난수 시드 고정으로 재현성 확보 (np.random.RandomState)
- fig/ 디렉토리에 결과 저장

---

## 실험 계획

### 정책 비교
| 정책 | 실험 | 구현 상태 | 클래스명 |
|------|------|----------|----------|
| Fixed Price | Baseline | ✅ | `FixedPricePolicy` |
| DP (Backward Induction) | 실험 1 | ✅ | `BackwardInductionPolicy` |
| Rollout (Monte Carlo) | 실험 2 | ✅ | `RolloutPolicy` |
| Greedy (Myopic) | 실험 2 | ✅ | `GreedyPolicy` |
| Random Price | (보류) | ✅ | `RandomPolicy` |
| Deterministic Lookahead | (보류) | ⬜ | 구현 예정 |
| Linear VFA | (보류) | ⬜ | 구현 예정 |
| UCB | (보류) | ⬜ | 구현 예정 |

### 실험 시나리오
| 시나리오 | 좌석 | 기간 | 상태 공간 | 목적 |
|----------|------|------|-----------|------|
| Small | 150 | 12 | 1,800 | 정확성 검증 |
| Medium | 300 | 30 | 9,000 | 스케일 테스트 |
| Large | 500 | 90 | 45,000 | DP vs Rollout 시간 비교 |

### 수요 모델 시나리오 (다항로짓 파라미터)

수요 함수: \( D_t = M_t \times \frac{e^{V_1 - \beta P_1}}{e^{V_1 - \beta P_1} + \delta(s_2) e^{V_2 - \beta P_2} + e^{V_0}} \)

- V: 점유율 기반 효용
- β: 가격 민감도
- V₀: 구매 포기 효용
- δ: 재고 함수

#### 시나리오 1: 인천-뉴욕 노선 (비즈니스 수요)
| 파라미터 | 값 | 비고 |
|----------|-----|------|
| P₁ (자사 점유율) | 0.608 | (1-0.05) × 0.64 |
| P₂ (경쟁사 점유율) | 0.342 | |
| β (가격 민감도) | 0.204 | 1.2 / (15.0 × 0.392) |
| V₁ (자사 효용) | 5.56 | ln(0.608/0.05) + 0.204×15.0 |
| V₂ (경쟁사 효용) | 4.98 | ln(0.342/0.05) + 0.204×15.0 |
| 가격 범위 | [10.0, 20.0] | 단위: 십만원, 평균 15.0 |
| 가격 탄력성 | -1.2 | 비즈니스 수요 (비탄력적) |
| 구매 포기 비율 | 5% | 비즈니스 노선 특성 |

#### 시나리오 2: 인천-호놀룰루 노선 (레저 수요)
| 파라미터 | 값 | 비고 |
|----------|-----|------|
| P₁ (자사 점유율) | 0.497 | (1-0.3) × (0.54/0.76) |
| P₂ (경쟁사 점유율) | 0.203 | |
| β (가격 민감도) | 0.191 | 0.6 / (8.0 × 0.392) |
| V₁ (자사 효용) | 2.04 | ln(0.497/0.3) + 0.191×8.0 |
| V₂ (경쟁사 효용) | 1.14 | ln(0.203/0.3) + 0.191×8.0 |
| 가격 범위 | [6.0, 10.0] | 단위: 십만원, 평균 8.0 |
| 가격 탄력성 | -0.6 | 레저 수요 (탄력적) |
| 구매 포기 비율 | 30% | 레저 노선 특성 |

**시나리오 특성 비교**:
- 비즈니스(시나리오1): 높은 가격대, 낮은 가격 탄력성, 낮은 포기율
- 레저(시나리오2): 낮은 가격대, 높은 가격 탄력성, 높은 포기율

### 비교 지표
1. Total Revenue (평균, 표준편차)
2. Running Time (정책 계산 시간)
3. Optimality Gap: `(DP수익 - Rollout수익) / DP수익`

### Rollout 설계
- Base Policy: Fixed Price 또는 Greedy
- Full Rollout (마지막 스테이지까지)
- 시뮬레이션 횟수 N: 100~500

---

## 구체적 실험 설계

### Baseline: Fixed Price

**목적**: 고정 가격 정책 성능 측정 (비교 기준)

| 정책 | 설명 |
|------|------|
| Fixed Price | 전 기간 동일 가격 유지 |

**실험 파라미터**:
```
시나리오 1: 인천-뉴욕 (fixed_price=15.0, 평균 가격)
시나리오 2: 인천-호놀룰루 (fixed_price=8.0, 평균 가격)
시뮬레이션 횟수: 1000회
```

**비교 지표**:
1. Mean Total Revenue
2. Std Total Revenue
3. 최종 잔여 좌석 분포

---

### 실험 1: DP (Backward Induction)

**목적**: Bellman 방정식 기반 Exact 최적 정책 계산

**방법론**:
- Backward Induction으로 V*(t, c) 테이블 계산
- V(t, c) = max_p { E[r(c, p, M) + γ·V(t+1, c')] }
- 모든 (t, c) 상태에 대해 최적 가격 π*(t, c) 저장

**핵심 특징**:
- Exact 최적 정책 (ground truth)
- 상태 공간 크기에 따라 계산 비용 증가
- 수요 모델(포아송, 로짓)을 정확히 알고 있다고 가정

**실험 파라미터**:
```
시나리오 1: 인천-뉴욕 (300석, 90기간, 상태공간 27,000)
시나리오 2: 인천-호놀룰루 (300석, 90기간, 상태공간 27,000)
할인율 γ: 1.0 (할인 없음)
시뮬레이션 횟수: 1000회
```

**비교 지표**:
1. Mean Total Revenue (Baseline 대비)
2. DP 계산 시간 (초)
3. 정책 테이블 시각화 (가격 히트맵)

---

### 실험 2: Lookahead (Rollout)

**목적**: Monte Carlo 시뮬레이션 기반 Approximate 정책 평가

**방법론**:
- 현재 상태에서 각 가격에 대해 N번 rollout 시뮬레이션
- 평균 수익이 가장 높은 가격 선택
- Base policy로 미래 스텝 시뮬레이션

**핵심 특징**:
- DP 대비 빠른 계산 (전체 테이블 계산 불필요)
- 시뮬레이션 횟수 N에 따라 정확도-속도 트레이드오프
- Base policy 선택에 따라 성능 변화

| 정책 | 설명 | 비교 포인트 |
|------|------|-------------|
| Rollout (Fixed base) | base_policy=FixedPrice | 단순 base |
| Rollout (Greedy base) | base_policy=Greedy | 더 좋은 base |
| Greedy (Myopic) | 즉시 수익만 최대화 | 미래 무시 baseline |

**실험 파라미터**:
```
시나리오 1: 인천-뉴욕 (300석, 90기간)
시나리오 2: 인천-호놀룰루 (300석, 90기간)
Rollout N: 100
Rollout base_policy: FixedPrice, Greedy
시뮬레이션 횟수: 1000회
```

**비교 지표**:
1. Mean Total Revenue
2. Optimality Gap: `(DP수익 - 정책수익) / DP수익 × 100%`
3. Policy 계산 시간 (초)

---

## (보류) 추가 실험

### (보류) 실험 3: VFA 스케일링

**목적**: 대규모 상태 공간에서 VFA 근사 성능 평가

| 방법론 | 형태 | 파라미터 수 |
|--------|------|-------------|
| DP (Exact) | 테이블 | nS × T |
| Linear VFA | V(c,t) ≈ θ₀ + θ₁·c + θ₂·t | 3 |
| Quadratic VFA | V(c,t) ≈ θ₀ + θ₁·c + θ₂·t + θ₃·c² + θ₄·c·t | 5 |

**구현 TODO**:
- [ ] `LinearVFA`, `QuadraticVFA` 클래스 추가 (policy.py)
- [ ] `fit_vfa()` 함수: DP 결과로 VFA 파라미터 학습
- [ ] `VFAPolicy` 클래스: VFA 기반 greedy 정책

---

### (보류) 실험 4: 온라인 학습 (UCB/IE)

**목적**: 수요 모델을 모를 때 탐험-활용 학습 성능 평가

**설정**: Multi-Armed Bandit (MAB)
- 각 가격 p_t를 arm으로 취급
- 가격별 기대 수익 E[r | p_t]를 직접 학습

| 정책 | 핵심 아이디어 | 수식 |
|------|---------------|------|
| UCB | 신뢰구간 상한 선택 | μ̂ + θ·√(log(t)/n) |
| Interval Estimation | 정밀도 기반 구간 | μ̂ + θ/√n |
| ε-Greedy | ε 확률로 탐험 | P(explore) = ε |

**구현 TODO**:
- [ ] `UCBPolicy` 클래스 추가 (policy.py)
- [ ] `IntervalEstimationPolicy` 클래스 추가
- [ ] `EpsilonGreedyPolicy` 클래스 추가

---

## 실험 실행 순서

```
Baseline: Fixed Price
├── 시나리오 1, 2에서 Fixed Price 시뮬레이션
└── 비교 기준 수치 확보

실험 1: DP (Backward Induction)
├── BackwardInductionPolicy.solve() 실행
├── 최적 정책 테이블 저장
└── 시뮬레이션 및 baseline 대비 비교

실험 2: Lookahead (Rollout)
├── Rollout (base=Fixed), Rollout (base=Greedy), Greedy 비교
├── DP 대비 Optimality Gap 계산
└── 계산 시간 비교

(보류) 실험 3, 4
```

---

## 결과 시각화 계획

| 실험 | 플롯 | 파일명 |
|------|------|--------|
| Baseline | Revenue distribution | `fig/baseline_revenue.png` |
| 1 (DP) | Policy heatmap (가격 vs 시간/좌석) | `fig/exp1_policy_heatmap.png` |
| 1 (DP) | Value function surface | `fig/exp1_value_function.png` |
| 2 (Rollout) | Bar chart: Revenue by policy | `fig/exp2_revenue_comparison.png` |
| 2 (Rollout) | Bar chart: Computation time | `fig/exp2_computation_time.png` |
| 종합 | 모든 정책 비교 테이블 | `fig/summary_comparison.png` |
