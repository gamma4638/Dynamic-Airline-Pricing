# -*- coding: utf-8 -*-
'''
Dynamic Airline Pricing MDP - Policy

모든 정책 클래스를 정의한다:
- BasePolicy: 추상 인터페이스
- FixedPricePolicy, RandomPolicy: 베이스라인
- GreedyPolicy: 근시안적 최적
- BackwardInductionPolicy: Bellman + DP (Exact 최적)
- RolloutPolicy: Monte Carlo 기반 근사
- DeterministicLookaheadPolicy: 기대값 기반 Lookahead
'''
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import poisson


class BasePolicy(ABC):
    """정책 인터페이스 (추상 클래스)"""

    @abstractmethod
    def select_action(self, state: int, t: int) -> float:
        """
        현재 상태와 시점에서 가격 선택

        Args:
            state: 현재 잔여 좌석 (c_t)
            t: 현재 시점

        Returns:
            p_t: 선택한 가격
        """
        raise NotImplementedError

    def reset(self):
        """에피소드 시작 시 정책 초기화 (필요한 경우)"""
        pass


class FixedPricePolicy(BasePolicy):
    """고정 가격 정책 (테스트/베이스라인용)"""

    def __init__(self, fixed_price: float = 500.0):
        """
        Args:
            fixed_price: 고정 가격
        """
        self.fixed_price = fixed_price

    def select_action(self, state: int, t: int) -> float:
        return self.fixed_price


class RandomPolicy(BasePolicy):
    """랜덤 가격 정책 (테스트용)"""

    def __init__(self, action_space: np.ndarray, seed: int = None):
        """
        Args:
            action_space: 가능한 가격 배열
            seed: 난수 시드
        """
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, t: int) -> float:
        return self.rng.choice(self.action_space)


# ========== Greedy 정책 ==========
class GreedyPolicy(BasePolicy):
    """
    Greedy (Myopic) 정책

    즉시 수익 E[p * sales]를 최대화하는 가격 선택.
    미래를 고려하지 않는 근시안적 최적 정책.
    """

    def __init__(self, model):
        """
        Args:
            model: AirlinePricingModel 인스턴스
        """
        self.model = model

    def select_action(self, state: int, t: int) -> float:
        """현재 스텝에서 즉시 기대 수익을 최대화하는 가격 선택"""
        if state == 0:
            return float(self.model.action_space[0])

        # 시점 t의 기대 경쟁사 가격 (노이즈 없이)
        p_comp = self.model.get_competitor_price(t, rng=None)

        best_action = None
        best_revenue = -np.inf

        # 시점 t의 기대 도착 수
        mu_t = self.model.get_expected_arrivals(t)

        for p_t in self.model.action_space:
            # 구매 확률 (동적 p_comp 사용)
            prob = self.model.purchase_probability(p_t, p_comp)
            # 기대 수요: E[Q] = mu_t * prob (시간별 mu)
            expected_demand = mu_t * prob
            # 기대 판매량 (좌석 제약)
            expected_sales = min(expected_demand, state)
            # 기대 수익
            expected_revenue = p_t * expected_sales

            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_action = p_t

        return float(best_action)


# ========== DP 기반 정책 ==========
class DPPolicy(BasePolicy):
    """미리 계산된 정책 테이블 기반 정책 (lookup only)"""

    def __init__(self, policy_table: np.ndarray):
        """
        Args:
            policy_table: 정책 테이블 (stage x state)
                          policy_table[t, s] = 시점 t, 잔여좌석 s에서의 가격
        """
        self.policy_table = policy_table

    def select_action(self, state: int, t: int) -> float:
        return float(self.policy_table[t, state])


class BackwardInductionPolicy(BasePolicy):
    """
    Bellman + Backward Induction으로 최적 정책 계산

    V(t, c) = max_p { E[r(c, p, M) + γ·V(t+1, c')] }

    where:
        r(c, p, M) = p · min(Q, c)
        Q = M · P(buy|p)
        M ~ Poisson(μ_t)  # 시간별 기대 도착 수
    """

    def __init__(self, model, gamma: float = 1.0, verbose: bool = True):
        """
        Args:
            model: AirlinePricingModel 인스턴스
            gamma: 할인율 (기본 1.0, 할인 없음)
            verbose: BI 진행 상황 출력 여부
        """
        self.model = model
        self.gamma = gamma
        self.verbose = verbose

        # 시간별 포아송 분포 사전 계산
        self._precompute_poisson_distributions()

        # 가치 함수 및 정책 테이블
        self.value = None
        self.policy_table = None
        self._solved = False

    def _precompute_poisson_distributions(self):
        """시간별 Poisson 분포 사전 계산"""
        T = self.model.num_stages

        # 각 시점의 기대 도착 수
        self.mu_by_t = [self.model.get_expected_arrivals(t) for t in range(T)]

        # 최대 M 값 (가장 큰 mu 기준)
        max_mu = max(self.mu_by_t) if self.mu_by_t else 1.0
        self.M_max = int(max_mu * 3 + 10)  # 3σ + 여유
        self.M_range = np.arange(0, self.M_max + 1)

        # 시간별 Poisson 확률
        self.poisson_probs_by_t = []
        for t in range(T):
            mu_t = self.mu_by_t[t]
            if mu_t > 0:
                probs = poisson.pmf(self.M_range, mu_t)
                probs /= probs.sum()  # 정규화
            else:
                probs = np.zeros(len(self.M_range))
                probs[0] = 1.0  # mu=0이면 M=0 확정
            self.poisson_probs_by_t.append(probs)

    def solve(self):
        """
        Backward Induction 실행

        시간 T-1부터 0까지 역순으로 최적 가치/정책 계산
        """
        nS = self.model.num_seats + 1
        T = self.model.num_stages

        self.value = np.zeros((T + 1, nS))
        self.policy_table = np.zeros((T, nS))

        # 역순으로 계산 (t = T-1, T-2, ..., 0)
        for t in range(T - 1, -1, -1):
            if self.verbose:
                print(f"  Solving t={t}/{T-1}")

            for c in range(nS):
                if c == 0:
                    # 좌석 없으면 가치 0
                    self.value[t, c] = 0.0
                    self.policy_table[t, c] = self.model.action_space[0]
                    continue

                # 모든 가격에 대해 기대 가치 계산
                best_value = -np.inf
                best_action = self.model.action_space[0]

                for p_t in self.model.action_space:
                    v = self._compute_expected_value(t, c, p_t)
                    if v > best_value:
                        best_value = v
                        best_action = p_t

                self.value[t, c] = best_value
                self.policy_table[t, c] = best_action

        self._solved = True
        if self.verbose:
            print("  Backward Induction completed.")

    def _compute_expected_value(self, t: int, c: int, p_t: float) -> float:
        """
        Bellman equation 계산: E[r + γV(t+1, c')]

        Args:
            t: 현재 시점
            c: 현재 잔여 좌석
            p_t: 선택한 가격

        Returns:
            기대 가치
        """
        # 시점 t의 기대 경쟁사 가격 (노이즈 없이)
        p_comp = self.model.get_competitor_price(t, rng=None)
        prob_buy = self.model.purchase_probability(p_t, p_comp)

        # 시점 t의 Poisson 분포 사용
        poisson_probs_t = self.poisson_probs_by_t[t]

        total_value = 0.0
        for M_t, prob_M in zip(self.M_range, poisson_probs_t):
            # 기대 수요
            Q_t = M_t * prob_buy
            # 실제 판매량 (좌석 제약)
            sold = min(int(np.round(Q_t)), c)
            # 즉시 보상
            reward = p_t * sold
            # 다음 상태
            c_next = c - sold
            # 미래 가치 (t+1의 value 사용)
            future_value = self.gamma * self.value[t + 1, c_next]
            # 기대값 누적
            total_value += prob_M * (reward + future_value)

        return total_value

    def select_action(self, state: int, t: int) -> float:
        """
        최적 가격 선택 (사전에 solve() 필요)
        """
        if not self._solved:
            raise RuntimeError("solve()를 먼저 호출하세요.")
        return float(self.policy_table[t, state])

    def get_value(self, state: int, t: int) -> float:
        """상태-시점의 가치 함수 반환"""
        if not self._solved:
            raise RuntimeError("solve()를 먼저 호출하세요.")
        return float(self.value[t, state])


# ========== Rollout 기반 정책 ==========
class RolloutPolicy(BasePolicy):
    """
    Rollout 기반 정책 (시뮬레이션 기반 의사결정)

    각 행동에 대해 N번의 시뮬레이션을 수행하고,
    평균 수익이 가장 높은 행동을 선택한다.
    """

    def __init__(self,
                 model,
                 base_policy: BasePolicy,
                 n_simulations: int = 100,
                 rollout_depth: int = 10,
                 seed: int = None):
        """
        Args:
            model: AirlinePricingModel 인스턴스
            base_policy: Rollout 중 사용할 기저 정책 (예: FixedPricePolicy)
            n_simulations: 각 행동당 시뮬레이션 횟수
            rollout_depth: 시뮬레이션 깊이 (K 스텝)
            seed: 난수 시드
        """
        self.model = model
        self.base_policy = base_policy
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, t: int) -> float:
        """
        현재 상태에서 최고 기대 수익의 가격 선택

        모든 가능한 가격에 대해 rollout 시뮬레이션을 수행하고,
        평균 수익이 가장 높은 가격을 반환한다.
        """
        if state == 0:
            return float(self.model.action_space[0])

        best_action = None
        best_value = -np.inf

        for action in self.model.action_space:
            value = self._evaluate_action(action, state, t)
            if value > best_value:
                best_value = value
                best_action = action

        return float(best_action)

    def _evaluate_action(self, action: float, state: int, t: int) -> float:
        """특정 행동의 기대값 계산 (N번 시뮬레이션 평균)"""
        total = 0.0
        for _ in range(self.n_simulations):
            total += self._simulate_rollout(state, t, action)
        return total / self.n_simulations

    def _simulate_rollout(self, state: int, t: int, first_action: float) -> float:
        """
        Truncated Rollout 시뮬레이션

        Args:
            state: 시작 상태 (잔여 좌석)
            t: 시작 시점
            first_action: 첫 번째 행동 (평가할 가격)

        Returns:
            total_reward: rollout_depth 스텝 동안의 총 수익
        """
        c_t = state
        total_reward = 0.0

        for step in range(self.rollout_depth):
            if c_t == 0:
                break

            current_t = t + step

            # 첫 스텝: 평가할 action 사용, 이후: base_policy 사용
            if step == 0:
                p_t = first_action
            else:
                p_t = self.base_policy.select_action(c_t, current_t)

            result = self.model.step(c_t, p_t, self.rng, t=current_t)
            total_reward += result['r_t']
            c_t = result['c_next']

        return total_reward
