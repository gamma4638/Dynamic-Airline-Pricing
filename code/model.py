# -*- coding: utf-8 -*-
'''
Dynamic Airline Pricing MDP - Model
basic_model.md 기반 구현
'''
import numpy as np


class AirlinePricingModel:
    """
    항공권 동적 가격 결정 MDP 환경

    State: c_t (잔여 좌석)
    Action: p_t (가격)
    Exogenous: M_t ~ Poisson(mu), 다항 로짓 수요 모델
    Transition: c_{t+1} = c_t - s_t
    Reward: r_t = p_t * s_t
    """

    def __init__(
        self,
        # State 관련
        num_seats: int = 150,
        num_stages: int = 90,
        # Action 관련
        price_min: float = 100.0,
        price_max: float = 1000.0,
        price_step: float = 100.0,
        # Demand Model 관련
        demand_multiplier: float = 2.5,
        demand_beta: float = 0.03,
        v_me: float = 5.0,
        v_comp: float = 4.5,
        v_no_buy: float = 1.0,
        beta: float = 0.01,
        p_comp_min: float = 5.0,
        p_comp_max: float = 15.0,
    ):
        """
        Args:
            num_seats: 총 좌석 수 (초기 c_0)
            num_stages: 판매 기간 T
            price_min: 최소 가격
            price_max: 최대 가격
            price_step: 가격 단위
            demand_multiplier: 지수 수요 모델의 총 수요 배수 (A = num_seats * multiplier)
            demand_beta: 지수 수요 모델의 감쇠율
            v_me: 자사 기본 효용
            v_comp: 경쟁사 기본 효용
            v_no_buy: 미구매 효용
            beta: 가격 민감도
            p_comp_min: 경쟁사 최소 가격 (t=0)
            p_comp_max: 경쟁사 최대 가격 (t=T-1)
        """
        # State 관련
        self.num_seats = num_seats
        self.num_stages = num_stages

        # Action 관련
        self.price_min = price_min
        self.price_max = price_max
        self.price_step = price_step

        # Demand Model 관련 (지수 수요 모델)
        self.demand_multiplier = demand_multiplier
        self.demand_beta = demand_beta
        self.v_me = v_me
        self.v_comp = v_comp
        self.v_no_buy = v_no_buy
        self.beta = beta
        self.p_comp_min = p_comp_min
        self.p_comp_max = p_comp_max

        # 지수 수요 모델: 일별 기대 도착 수 사전 계산
        self._precompute_daily_demand()

        # State Space: [0, 1, ..., num_seats]
        self.state_space = list(range(num_seats + 1))

        # Action Space: [price_min, price_min + step, ..., price_max]
        self.action_space = np.arange(price_min, price_max + 1e-3, price_step)

    def _precompute_daily_demand(self):
        """
        지수 수요 모델: 일별 기대 도착 수 사전 계산

        cumulative(t) = A * exp(-demand_beta * (T - t))
        daily_demand[t] = cumulative(t+1) - cumulative(t)

        t=0이 판매 시작, t=T-1이 출발 직전 (수요가 가장 높음)
        """
        T = self.num_stages
        A = self.num_seats * self.demand_multiplier

        # t=0 (T일 전) ~ t=T (출발일)
        # 남은 일수: T-t (t=0이면 T일 남음, t=T이면 0일 남음)
        t_remaining = np.arange(T, -1, -1)  # [T, T-1, ..., 1, 0]
        cumulative = A * np.exp(-self.demand_beta * t_remaining)

        # 일별 수요: cumulative[t+1] - cumulative[t]
        self.daily_demand = np.diff(cumulative)  # 길이: T

    def get_expected_arrivals(self, t: int) -> float:
        """
        시점 t의 기대 도착 고객 수 반환

        Args:
            t: 현재 시점 (0 ~ num_stages-1)

        Returns:
            기대 도착 고객 수 (M_t의 기대값)
        """
        if t < 0 or t >= self.num_stages:
            return 0.0
        return float(self.daily_demand[t])

    # ========== Competitor Price (동적 경쟁사 가격) ==========
    def get_competitor_price(self, t: int, rng: np.random.Generator = None) -> float:
        """
        시간 t에 따른 경쟁사 가격 반환

        기본 가격: 선형 증가 (p_comp_min → p_comp_max)
        노이즈: rng가 주어지면 Exponential 노이즈 추가

        Args:
            t: 현재 시점 (0 ~ num_stages-1)
            rng: 난수 생성기 (None이면 기대값만 반환)

        Returns:
            경쟁사 가격 (p_comp_min ~ p_comp_max 범위)
        """
        # 선형 증가: t=0에서 min, t=T-1에서 max
        base = self.p_comp_min + (self.p_comp_max - self.p_comp_min) * (t / max(1, self.num_stages - 1))

        if rng is not None:
            # 평균 0인 노이즈 (exponential - mean)
            noise = rng.exponential(scale=0.5) - 0.5
            price = base + noise
        else:
            price = base

        return float(np.clip(price, self.p_comp_min, self.p_comp_max))

    # ========== Exogenous Information (Demand Model) ==========
    def sample_arrivals(self, rng: np.random.Generator, t: int = None) -> int:
        """
        도착 고객 수 샘플링 (Poisson with time-varying mean)

        Args:
            rng: 난수 생성기
            t: 현재 시점 (None이면 전체 평균 사용)

        Returns:
            도착 고객 수 M_t ~ Poisson(mu_t)
        """
        if t is not None:
            mu_t = self.get_expected_arrivals(t)
        else:
            # 하위 호환성: 전체 평균
            mu_t = np.mean(self.daily_demand)
        return rng.poisson(mu_t)

    def purchase_probability(self, p_t: float, p_comp: float = None) -> float:
        """
        다항 로짓 모델: 자사 구매 확률 계산
        P(buy_me) = exp(U_me) / (exp(U_me) + exp(U_comp) + exp(U_no_buy))

        Args:
            p_t: 자사 가격
            p_comp: 경쟁사 가격 (None이면 p_comp_min과 p_comp_max의 중간값 사용)
        """
        if p_comp is None:
            # 하위 호환성: 기본값은 중간 가격
            p_comp = (self.p_comp_min + self.p_comp_max) / 2

        U_me = self.v_me - self.beta * p_t
        U_comp = self.v_comp - self.beta * p_comp
        U_no_buy = self.v_no_buy

        # 수치 안정성을 위한 max 빼기
        max_U = max(U_me, U_comp, U_no_buy)
        exp_me = np.exp(U_me - max_U)
        exp_comp = np.exp(U_comp - max_U)
        exp_no = np.exp(U_no_buy - max_U)

        return exp_me / (exp_me + exp_comp + exp_no)

    def compute_demand(self, p_t: float, M_t: int, p_comp: float = None) -> float:
        """수요 계산: Q_t = M_t * P(buy_me)"""
        prob = self.purchase_probability(p_t, p_comp)
        return M_t * prob

    # ========== Transition Function ==========
    def compute_sales(self, Q_t: float, c_t: int) -> int:
        """실제 판매량: s_t = min(round(Q_t), c_t)"""
        return min(int(round(Q_t)), c_t)

    def transition(self, c_t: int, s_t: int) -> int:
        """상태 전이: c_{t+1} = c_t - s_t"""
        return c_t - s_t

    # ========== Objective Function ==========
    def compute_reward(self, p_t: float, s_t: int) -> float:
        """즉시 보상: r_t = p_t * s_t"""
        return p_t * s_t

    @staticmethod
    def compute_total_revenue(rewards: list) -> float:
        """총 매출: J = sum(r_t), 할인 없음"""
        return sum(rewards)

    # ========== Step (편의 메서드) ==========
    def step(self, c_t: int, p_t: float, rng: np.random.Generator, t: int = None) -> dict:
        """
        한 스텝 실행

        Args:
            c_t: 현재 잔여 좌석
            p_t: 선택한 가격
            rng: 난수 생성기
            t: 현재 시점 (동적 p_comp 계산용, None이면 기본 p_comp 사용)

        Returns:
            dict: M_t, Q_t, s_t, r_t, c_next, p_comp
        """
        # 동적 경쟁사 가격 계산
        if t is not None:
            p_comp = self.get_competitor_price(t, rng)
        else:
            p_comp = (self.p_comp_min + self.p_comp_max) / 2

        # 시간별 기대 도착 수로 샘플링
        M_t = self.sample_arrivals(rng, t)
        Q_t = self.compute_demand(p_t, M_t, p_comp)
        s_t = self.compute_sales(Q_t, c_t)
        r_t = self.compute_reward(p_t, s_t)
        c_next = self.transition(c_t, s_t)

        return {
            'M_t': M_t,
            'Q_t': Q_t,
            's_t': s_t,
            'r_t': r_t,
            'c_next': c_next,
            'p_comp': p_comp,
        }
