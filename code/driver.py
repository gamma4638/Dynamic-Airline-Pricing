# -*- coding: utf-8 -*-
'''
Dynamic Airline Pricing MDP - Driver
시뮬레이션 실행
'''
import os
import argparse
import numpy as np

from model import AirlinePricingModel
from policy import (
    BasePolicy,
    FixedPricePolicy,
    RandomPolicy,
    GreedyPolicy,
    BackwardInductionPolicy,
    RolloutPolicy,
)
from plot import plot_episode, plot_statistics, plot_policy_comparison, plot_policy_3d, plot_policy_heatmap


class Simulator:
    """시뮬레이션 드라이버"""

    def __init__(self, model: AirlinePricingModel, policy: BasePolicy, seed: int = None):
        """
        Args:
            model: MDP 환경
            policy: 정책
            seed: 난수 시드
        """
        self.model = model
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def run_episode(self) -> dict:
        """
        한 에피소드 실행

        Returns:
            dict: 에피소드 결과
                - states: 각 시점 잔여 좌석 [c_0, c_1, ..., c_T]
                - actions: 각 시점 가격 [p_0, p_1, ..., p_{T-1}]
                - rewards: 각 시점 수익 [r_0, r_1, ..., r_{T-1}]
                - arrivals: 각 시점 도착 고객 [M_0, M_1, ..., M_{T-1}]
                - sales: 각 시점 판매량 [s_0, s_1, ..., s_{T-1}]
                - total_revenue: 총 매출
        """
        self.policy.reset()

        states = [self.model.num_seats]  # c_0
        actions = []
        rewards = []
        arrivals = []
        sales = []

        c_t = self.model.num_seats

        for t in range(self.model.num_stages):
            if c_t == 0:
                # 좌석 소진 시 남은 기간은 0으로 채움
                actions.append(0.0)
                rewards.append(0.0)
                arrivals.append(0)
                sales.append(0)
                states.append(0)
                continue

            # 정책에서 가격 선택
            p_t = self.policy.select_action(c_t, t)
            actions.append(p_t)

            # 환경 스텝 실행
            result = self.model.step(c_t, p_t, self.rng)

            arrivals.append(result['M_t'])
            sales.append(result['s_t'])
            rewards.append(result['r_t'])
            c_t = result['c_next']
            states.append(c_t)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'arrivals': np.array(arrivals),
            'sales': np.array(sales),
            'total_revenue': sum(rewards),
        }

    def run_multiple(self, n_episodes: int, verbose: bool = True) -> dict:
        """
        여러 에피소드 실행 및 통계 계산

        Args:
            n_episodes: 에피소드 수
            verbose: 진행 상황 출력 여부

        Returns:
            dict: 통계 결과
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_arrivals = []
        all_sales = []
        total_revenues = []

        for i in range(n_episodes):
            if verbose and (i + 1) % 100 == 0:
                print(f"Episode {i + 1}/{n_episodes}")

            result = self.run_episode()
            all_states.append(result['states'])
            all_actions.append(result['actions'])
            all_rewards.append(result['rewards'])
            all_arrivals.append(result['arrivals'])
            all_sales.append(result['sales'])
            total_revenues.append(result['total_revenue'])

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_rewards = np.array(all_rewards)
        all_arrivals = np.array(all_arrivals)
        all_sales = np.array(all_sales)
        total_revenues = np.array(total_revenues)

        return {
            # Raw data
            'all_states': all_states,
            'all_actions': all_actions,
            'all_rewards': all_rewards,
            'all_arrivals': all_arrivals,
            'all_sales': all_sales,
            'total_revenues': total_revenues,
            # Statistics
            'mean_states': np.mean(all_states, axis=0),
            'mean_actions': np.mean(all_actions, axis=0),
            'mean_rewards': np.mean(all_rewards, axis=0),
            'mean_revenue': np.mean(total_revenues),
            'std_revenue': np.std(total_revenues),
            'min_revenue': np.min(total_revenues),
            'max_revenue': np.max(total_revenues),
        }


# ========== 메인 실행 ==========
def main():
    parser = argparse.ArgumentParser(
        description='Dynamic Airline Pricing Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python driver.py --scenario 1 --mode single           # 시나리오1(뉴욕) 단일 에피소드
  python driver.py --scenario 2 --mode multiple -p dp   # 시나리오2(호놀룰루) DP 정책
  python driver.py --mode multiple -p dp                # DP 정책으로 1000회 시뮬레이션
  python driver.py --mode single -p greedy              # Greedy 정책 단일 에피소드
  python driver.py --mode multiple -p rollout -n 500    # Rollout 정책 500회
  python driver.py -p fixed --fixed-price 12            # Fixed Price 12(십만원)로 설정
        '''
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['single', 'multiple', 'both'],
        default='multiple',
        help='실행 모드: single(단일), multiple(다중), both(둘 다) (기본: single)'
    )
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=100,
        help='multiple 모드에서 에피소드 수 (기본: 1000)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='랜덤 시드 (기본: 42)'
    )
    parser.add_argument(
        '--policy', '-p',
        type=str,
        choices=['fixed', 'random', 'greedy', 'dp', 'rollout'],
        default='dp',
        help='정책 선택: fixed, random, greedy, dp, rollout (기본: fixed)'
    )
    parser.add_argument(
        '--fixed-price',
        type=float,
        default=None,
        help='FixedPricePolicy의 가격 (기본: 시나리오별 평균 가격)'
    )
    parser.add_argument(
        '--scenario',
        type=int,
        choices=[1, 2],
        default=1,
        help='시나리오 선택: 1(인천-뉴욕, 비즈니스), 2(인천-호놀룰루, 레저) (기본: 1)'
    )
    args = parser.parse_args()

    # 저장 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, '..', 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    # ===== 파라미터 설정 =====
    # 시나리오 1: 인천-뉴욕 노선 (비즈니스 수요)
    # - 높은 가격대, 낮은 가격 탄력성(-1.2), 낮은 구매 포기율(5%)
    model_params1 = {
        'num_seats': 600,
        'num_stages': 90,
        'price_min': 8.0,      # 단위: 십만원
        'price_max': 18.0,
        'price_step': 1.0,
        'demand_multiplier': 2.5,  # 지수 수요 모델: A = seats * multiplier
        'demand_beta': 0.03,       # 지수 수요 모델: 감쇠율
        'v_me': 5.56,           # V₁: 자사 효용
        'v_comp': 4.98,         # V₂: 경쟁사 효용
        'v_no_buy': 0.0,        # V₀: 구매 포기 효용 (기준점)
        'beta': 0.204,          # 가격 민감도
        'p_comp_min': 5.0,      # 경쟁사 최소 가격 (t=0)
        'p_comp_max': 15.0,     # 경쟁사 최대 가격 (t=T-1)
    }

    # 시나리오 2: 인천-호놀룰루 노선 (레저 수요)
    # - 낮은 가격대, 높은 가격 탄력성(-0.6), 높은 구매 포기율(30%)
    model_params2 = {
        'num_seats': 600,
        'num_stages': 90,
        'price_min': 6.0,       # 단위: 십만원
        'price_max': 10.0,
        'price_step': 0.5,
        'demand_multiplier': 2.5,  # 지수 수요 모델: A = seats * multiplier
        'demand_beta': 0.03,       # 지수 수요 모델: 감쇠율
        'v_me': 2.04,           # V₁: 자사 효용
        'v_comp': 1.14,         # V₂: 경쟁사 효용
        'v_no_buy': 0.0,        # V₀: 구매 포기 효용 (기준점)
        'beta': 0.191,          # 가격 민감도
        'p_comp_min': 5.0,      # 경쟁사 최소 가격 (t=0)
        'p_comp_max': 15.0,     # 경쟁사 최대 가격 (t=T-1)
    }

    # 시나리오 선택
    if args.scenario == 1:
        model_params = model_params1
        scenario_name = "시나리오1: 인천-뉴욕 (비즈니스)"
        default_fixed_price = 15.0  # 평균 가격
    else:
        model_params = model_params2
        scenario_name = "시나리오2: 인천-호놀룰루 (레저)"
        default_fixed_price = 8.0   # 평균 가격

    # fixed_price 기본값 설정
    if args.fixed_price is None:
        args.fixed_price = default_fixed_price

    # ===== 모델 생성 =====
    model = AirlinePricingModel(**model_params)

    print("=" * 50)
    print(f"[{scenario_name}]")
    print("Model Parameters:")
    print(f"  Seats: {model.num_seats}, Stages: {model.num_stages}")
    print(f"  Price Range: {model.price_min} ~ {model.price_max} (십만원)")
    print(f"  Action Space: {model.action_space}")
    print(f"  V_me: {model.v_me}, V_comp: {model.v_comp}, V_no_buy: {model.v_no_buy}")
    print(f"  Beta (가격 민감도): {model.beta}")
    print(f"  Demand Model: exponential (multiplier={model.demand_multiplier}, beta={model.demand_beta})")
    print(f"  Expected Arrivals: t=0: {model.get_expected_arrivals(0):.1f}, t=89: {model.get_expected_arrivals(89):.1f}")
    print(f"  Competitor Price: {model.p_comp_min} ~ {model.p_comp_max} (동적)")
    print("=" * 50)

    # ===== 정책 생성 =====
    print(f"\n[Policy: {args.policy.upper()}]")

    if args.policy == 'fixed':
        policy = FixedPricePolicy(fixed_price=args.fixed_price)
        policy_name = f"Fixed({args.fixed_price})"

    elif args.policy == 'random':
        policy = RandomPolicy(action_space=model.action_space, seed=args.seed)
        policy_name = "Random"

    elif args.policy == 'greedy':
        policy = GreedyPolicy(model=model)
        policy_name = "Greedy"

    elif args.policy == 'dp':
        print("  Running Backward Induction...")
        policy = BackwardInductionPolicy(model=model, gamma=1.0, verbose=True)
        policy.solve()
        policy_name = "DP (Backward Induction)"
        print(f"  Initial Value V(0, {model.num_seats}): {policy.get_value(model.num_seats, 0):,.0f}")

    elif args.policy == 'rollout':
        base_policy = FixedPricePolicy(fixed_price=args.fixed_price)
        policy = RolloutPolicy(
            model=model,
            base_policy=base_policy,
            n_simulations=100,
            rollout_depth=model.num_stages,
            seed=args.seed
        )
        policy_name = f"Rollout(base=Fixed({args.fixed_price}))"

    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # ===== 시뮬레이션 실행 =====
    simulator = Simulator(model, policy, seed=args.seed)

    # 단일 에피소드
    if args.mode in ['single', 'both']:
        print("\n[Single Episode]")
        episode_result = simulator.run_episode()
        print(f"  Total Revenue: {episode_result['total_revenue']:,.0f}")
        print(f"  Final Seats: {episode_result['states'][-1]}")

        plot_episode(
            episode_result,
            title=f"Single Episode ({policy_name})",
            save_path=os.path.join(fig_dir, 'episode_result.png')
        )

    # 여러 에피소드
    if args.mode in ['multiple', 'both']:
        print("\n[Multiple Episodes]")
        stats = simulator.run_multiple(args.episodes, verbose=True)
        print(f"  Mean Revenue: {stats['mean_revenue']:,.0f}")
        print(f"  Std Revenue: {stats['std_revenue']:,.0f}")
        print(f"  Min Revenue: {stats['min_revenue']:,.0f}")
        print(f"  Max Revenue: {stats['max_revenue']:,.0f}")

        plot_statistics(
            stats,
            title=f"Simulation Statistics ({args.episodes} episodes, {policy_name})",
            save_path=os.path.join(fig_dir, 'simulation_stats.png')
        )

    # DP 정책인 경우 정책 테이블 시각화
    if args.policy == 'dp':
        print("\n[Policy Visualization]")
        plot_policy_3d(
            policy.policy_table,
            title=f"Optimal Policy 3D ({policy_name})",
            save_path=os.path.join(fig_dir, 'policy_3d.png')
        )
        plot_policy_heatmap(
            policy.policy_table,
            title=f"Optimal Policy Heatmap ({policy_name})",
            save_path=os.path.join(fig_dir, 'policy_heatmap.png')
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
