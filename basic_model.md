# 모델링

### State Variable $S_t$

- $c_t$ : 시점 t 잔여좌석
- $x_{j}$ : 항공편 노선 j 에 대한 상수 (Optional)

### Decision Variable $X_t$

<aside>

결정해야 하는 것

- $A(t)$ 밴드의 평균 값 → 항공권 평균 가격 활용
</aside>

- $p_t\in A(t)$
    - Discrete, Finite Action Space : ex) [100, 200, 300, … 1000]

### Exogenous Information $W_{t+1}$ ⇒ Demand Model

<aside>

**결정해야 하는 것**

- $\tilde{M}_t$ : 그날 구매 관심 수요 ⇒ 포아송
- $p_\text{comp}$ : 경쟁사 가격 ⇒ 이산적? 분포 샘플링?
- $V_\text{me}, V_\text{comp}$ : 경쟁사 및 자사 기본 효용
- $V_0$ : 미구매 효용
- $\beta$ : 민감도 상수 결정 (자/타사 구분?)
</aside>

- $\tilde{M}_t$ : 오늘 도착자(예약자 수)
    - $\tilde{M}_t \sim \text{Poisson}(\mu_t)$ 에서 샘플링

- $Q_t(p_t, p_{\text{comp}}) = \tilde{M_t} \times \frac{\exp(U_{\text{me}})}{\exp(U_{\text{me}}) + \exp(U_{\text{comp}}) + \exp(U_{\text{no\_buy}})}$
    - $U_\text{me}$  : $V-\beta*p_t$ ⇒ 내 효용
    - $U_\text{comp} : V-\beta*p_\text{comp}$ ⇒ 경쟁사 효용
    - $U_\text{no\_buy}: \text{const}$
        - $V$ : 항공사 브랜드 서비스의 기본 가격
        - $\beta$ : 가격 민감도
        - 경쟁사 가격은 외생변수나 확률변수로 설정

- 들어온 사람 수 → 수요와 매핑
    - $s_t=\min(Q_t, c_t)$ :

### Transition Function  $S_{t+1}=f(X_t, S_t, W_{t+1})$

- 다음 시점 t 잔여좌석 = 직전 시점 t 잔여좌석 - 판매좌석

$$
c_{t+1}=c_t-s_t
$$

### Objective Function

- 시점 t 판매수익

$$
r_t=p_t*\min(Q_t,c_t)
$$

- 총 매출 합 (할인 X)

$$
J(\pi) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T-1} r_t\right]      = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T-1} p_t \cdot \min(Q_t, c_t)\right]
$$