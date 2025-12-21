# -*- coding: utf-8 -*-
'''
Dynamic Airline Pricing MDP - Plot
시뮬레이션 결과 시각화
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만 하는 백엔드
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12


def plot_episode(result: dict, title: str = "Single Episode", save_path: str = None):
    """
    단일 에피소드 결과 시각화

    Args:
        result: Simulator.run_episode() 반환값
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    T = len(result['actions'])
    time = np.arange(T)

    # 잔여 좌석
    axes[0, 0].plot(result['states'], marker='o', markersize=2)
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Remaining Seats')
    axes[0, 0].set_title('Remaining Seats over Time')
    axes[0, 0].grid(True)

    # 가격
    axes[0, 1].plot(time, result['actions'], marker='o', markersize=2)
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].set_title('Price over Time')
    axes[0, 1].grid(True)

    # 판매량
    axes[1, 0].bar(time, result['sales'], alpha=0.7)
    axes[1, 0].set_xlabel('Time (t)')
    axes[1, 0].set_ylabel('Sales')
    axes[1, 0].set_title('Sales over Time')
    axes[1, 0].grid(True)

    # 수익
    axes[1, 1].bar(time, result['rewards'], alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Time (t)')
    axes[1, 1].set_ylabel('Revenue')
    axes[1, 1].set_title(f'Revenue over Time (Total: {result["total_revenue"]:,.0f})')
    axes[1, 1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_statistics(stats: dict, title: str = "Simulation Statistics", save_path: str = None):
    """
    여러 에피소드 통계 시각화

    Args:
        stats: Simulator.run_multiple() 반환값
        title: 그래프 제목
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    T = len(stats['mean_actions'])
    time = np.arange(T)

    # 평균 잔여 좌석
    axes[0, 0].plot(stats['mean_states'], marker='o', markersize=2)
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Avg Remaining Seats')
    axes[0, 0].set_title('Average Remaining Seats')
    axes[0, 0].grid(True)

    # 평균 가격
    axes[0, 1].plot(time, stats['mean_actions'], marker='o', markersize=2)
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Avg Price')
    axes[0, 1].set_title('Average Price')
    axes[0, 1].grid(True)
    # y축 오프셋 표기 끄고 실제 값 표시, 0부터 시작하여 전체 범위 보이도록
    axes[0, 1].ticklabel_format(useOffset=False, style='plain', axis='y')
    axes[0, 1].set_ylim(0, np.max(stats['mean_actions']) * 1.15)

    # 평균 수익
    axes[1, 0].bar(time, stats['mean_rewards'], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Time (t)')
    axes[1, 0].set_ylabel('Avg Revenue')
    axes[1, 0].set_title('Average Revenue per Stage')
    axes[1, 0].grid(True)

    # 총 매출 분포
    axes[1, 1].hist(stats['total_revenues'], bins=30, alpha=0.7, color='blue')
    axes[1, 1].axvline(stats['mean_revenue'], color='red', linestyle='--',
                       label=f'Mean: {stats["mean_revenue"]:,.0f}')
    axes[1, 1].set_xlabel('Total Revenue')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Total Revenue Distribution (std: {stats["std_revenue"]:,.0f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_policy_comparison(results: dict, title: str = "Policy Comparison", save_path: str = None):
    """
    여러 정책 비교 시각화

    Args:
        results: {policy_name: stats} 딕셔너리
        title: 그래프 제목
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    policy_names = list(results.keys())
    mean_revenues = [results[name]['mean_revenue'] for name in policy_names]
    std_revenues = [results[name]['std_revenue'] for name in policy_names]

    # 평균 매출 비교
    x = np.arange(len(policy_names))
    bars = axes[0].bar(x, mean_revenues, yerr=std_revenues, capsize=5, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policy_names, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Total Revenue')
    axes[0].set_title('Mean Revenue by Policy')
    axes[0].grid(True, axis='y')

    # 매출 분포 비교
    for name in policy_names:
        axes[1].hist(results[name]['total_revenues'], bins=30, alpha=0.5, label=name)
    axes[1].set_xlabel('Total Revenue')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Revenue Distribution by Policy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_policy_heatmap(policy_table: np.ndarray, title: str = "Policy Heatmap",
                        save_path: str = None):
    """
    정책 테이블 히트맵 시각화

    Args:
        policy_table: (T, nS) 배열, policy_table[t, c] = 가격
        title: 그래프 제목
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # c=0 (좌석 없음) 제외: 좌석이 0일 때는 판매 불가하므로 의미 없는 가격
    policy_to_plot = policy_table[:, 1:].copy()  # c >= 1인 부분만

    im = ax.imshow(policy_to_plot.T, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Remaining Seats (c)')
    ax.set_title(title)

    # y축 레이블을 1부터 시작하도록 조정
    num_ticks = 5
    T, nS_minus1 = policy_to_plot.shape
    tick_positions = np.linspace(0, nS_minus1 - 1, num_ticks).astype(int)
    tick_labels = tick_positions + 1  # c=1부터 시작
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Price')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_policy_3d(policy_table: np.ndarray, title: str = "Policy 3D Surface",
                   save_path: str = None):
    """
    정책 테이블 3D surface 시각화

    Args:
        policy_table: (T, nS) 배열, policy_table[t, c] = 가격
        title: 그래프 제목
        save_path: 저장 경로
    """
    from mpl_toolkits.mplot3d import Axes3D

    T, nS = policy_table.shape

    # c=0 (좌석 없음) 제외: 좌석이 0일 때는 판매 불가하므로 의미 없는 가격
    policy_to_plot = policy_table[:, 1:].copy()  # c >= 1인 부분만
    t_range = np.arange(T)
    c_range = np.arange(1, nS)  # 1부터 시작
    T_grid, C_grid = np.meshgrid(t_range, c_range)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T_grid, C_grid, policy_to_plot.T,
                           cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Remaining Seats (c)')
    ax.set_zlabel('Price')
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Price')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def show_policy_3d_interactive(policy_table: np.ndarray, title: str = "Policy 3D Surface"):
    """
    정책 테이블 3D surface 인터랙티브 시각화
    - 마우스 드래그로 회전
    - GUI 창에서 직접 PNG 저장 가능 (툴바의 저장 버튼)

    Args:
        policy_table: (T, nS) 배열, policy_table[t, c] = 가격
        title: 그래프 제목
    """
    plt.switch_backend('TkAgg')  # GUI 백엔드로 전환
    from mpl_toolkits.mplot3d import Axes3D

    T, nS = policy_table.shape

    # c=0 (좌석 없음) 제외
    policy_to_plot = policy_table[:, 1:].copy()
    t_range = np.arange(T)
    c_range = np.arange(1, nS)
    T_grid, C_grid = np.meshgrid(t_range, c_range)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T_grid, C_grid, policy_to_plot.T,
                           cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Remaining Seats (c)')
    ax.set_zlabel('Price')
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Price')

    plt.tight_layout()
    print("Interactive 3D plot opened. Drag to rotate, use toolbar to save.")
    plt.show()
