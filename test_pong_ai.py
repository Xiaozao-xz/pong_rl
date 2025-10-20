#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pong_ai.py
测试训练好的主动推理乒乓球Agent
"""

import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from pong_env_ai import PongEnvActiveInference
from agent_cl import cl_agent


def load_agent(filepath, env):
    """加载训练好的agent"""
    print(f"正在加载agent: {filepath}")
    
    with open(filepath, 'rb') as f:
        agent_data = pickle.load(f)
    
    # 重建agent
    agent = cl_agent(
        A=agent_data['A'],
        B=agent_data['B'],
        C=agent_data['C'],
        D=agent_data['D'],
        action_precision=agent_data['action_precision'],
        planning_precision=agent_data['planning_precision'],
        memory_horizon=agent_data['memory_horizon'],
        gamma_initial=agent_data['gamma_initial']
    )
    
    # 恢复学习到的策略
    agent.CL = agent_data['CL']
    agent.Gamma = agent_data['Gamma']
    
    print("Agent加载成功!")
    return agent


def test_agent(env, agent, num_episodes=20, render=False, verbose=True):
    """
    测试训练好的agent
    
    Args:
        env: 环境
        agent: 训练好的agent
        num_episodes: 测试轮数
        render: 是否显示游戏画面
        verbose: 是否详细输出
    
    Returns:
        test_results: 测试结果字典
    """
    print("\n" + "=" * 70)
    print("开始测试Agent")
    print("=" * 70)
    
    test_results = {
        'rewards': [],
        'steps': [],
        'success_episodes': [],
        'episode_details': []
    }
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        trajectory = []
        
        while not done:
            # Agent决策 (不学习)
            action = agent.step(obs, learning=False)
            paddle_action = int(action[2]) if len(action) > 2 else 1
            
            # 记录轨迹
            trajectory.append({
                'step': episode_steps,
                'obs': obs,
                'action': paddle_action
            })
            
            obs, reward, done, info = env.step(paddle_action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                time.sleep(0.05)  # 放慢显示
        
        test_results['rewards'].append(episode_reward)
        test_results['steps'].append(episode_steps)
        test_results['episode_details'].append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'trajectory': trajectory
        })
        
        if episode_reward > 0:
            success_count += 1
            test_results['success_episodes'].append(episode + 1)
        
        if verbose:
            status = "✓ 成功" if episode_reward > 0 else "✗ 失败"
            print(f"Episode {episode + 1:2d}/{num_episodes} | "
                  f"Reward: {episode_reward:5.2f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"{status}")
    
    # 计算统计结果
    test_results['statistics'] = {
        'avg_reward': np.mean(test_results['rewards']),
        'std_reward': np.std(test_results['rewards']),
        'max_reward': np.max(test_results['rewards']),
        'min_reward': np.min(test_results['rewards']),
        'avg_steps': np.mean(test_results['steps']),
        'success_count': success_count,
        'success_rate': success_count / num_episodes
    }
    
    return test_results


def print_test_summary(test_results):
    """打印测试总结"""
    stats = test_results['statistics']
    
    print("\n" + "=" * 70)
    print("测试结果总结")
    print("=" * 70)
    
    print(f"\n总测试轮数: {len(test_results['rewards'])}")
    
    print(f"\n奖励统计:")
    print(f"  平均奖励: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  最大奖励: {stats['max_reward']:.2f}")
    print(f"  最小奖励: {stats['min_reward']:.2f}")
    
    print(f"\n步数统计:")
    print(f"  平均步数: {stats['avg_steps']:.1f}")
    
    print(f"\n成功统计:")
    print(f"  成功次数: {stats['success_count']}/{len(test_results['rewards'])}")
    print(f"  成功率: {stats['success_rate']:.2%}")
    
    if len(test_results['success_episodes']) > 0:
        print(f"  成功轮次: {test_results['success_episodes']}")
    
    print("=" * 70)


def visualize_test_results(test_results, save_dir='./figures'):
    """可视化测试结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    rewards = test_results['rewards']
    steps = test_results['steps']
    stats = test_results['statistics']
    
    # 奖励柱状图
    colors = ['green' if r > 0 else 'red' for r in rewards]
    axes[0, 0].bar(range(1, len(rewards) + 1), rewards, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].axhline(y=stats['avg_reward'], color='blue', linestyle='--', 
                       linewidth=2, label=f"平均: {stats['avg_reward']:.2f}")
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 步数柱状图
    axes[0, 1].bar(range(1, len(steps) + 1), steps, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=stats['avg_steps'], color='red', linestyle='--', 
                       linewidth=2, label=f"平均: {stats['avg_steps']:.1f}")
    axes[0, 1].set_title('Episode Steps', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 奖励分布
    axes[1, 0].hist(rewards, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(stats['avg_reward'], color='red', linestyle='--', 
                       linewidth=2, label=f"均值: {stats['avg_reward']:.2f}")
    axes[1, 0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 成功/失败饼图
    success_count = stats['success_count']
    fail_count = len(rewards) - success_count
    
    sizes = [success_count, fail_count]
    labels = [f'成功 ({success_count})', f'失败 ({fail_count})']
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    
    axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    axes[1, 1].set_title('Success vs Failure', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f'test_results_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n测试结果图已保存到: {filepath}")
    
    plt.show()


def compare_episodes(test_results, episode_indices=None):
    """比较特定轮次的详细信息"""
    if episode_indices is None:
        # 默认比较第一个成功和第一个失败的轮次
        success_idx = None
        failure_idx = None
        
        for i, reward in enumerate(test_results['rewards']):
            if reward > 0 and success_idx is None:
                success_idx = i
            if reward < 0 and failure_idx is None:
                failure_idx = i
            if success_idx is not None and failure_idx is not None:
                break
        
        episode_indices = [idx for idx in [success_idx, failure_idx] if idx is not None]
    
    if not episode_indices:
        print("没有找到可比较的轮次")
        return
    
    print("\n" + "=" * 70)
    print("轮次详细比较")
    print("=" * 70)
    
    for idx in episode_indices:
        detail = test_results['episode_details'][idx]
        print(f"\nEpisode {detail['episode']}:")
        print(f"  奖励: {detail['reward']:.2f}")
        print(f"  步数: {detail['steps']}")
        print(f"  结果: {'成功 ✓' if detail['reward'] > 0 else '失败 ✗'}")
        
        # 显示前几步的动作
        print(f"  前5步动作:")
        for step_info in detail['trajectory'][:5]:
            action_names = ['向上', '不动', '向下']
            print(f"    Step {step_info['step']}: "
                  f"obs={step_info['obs']}, "
                  f"action={action_names[step_info['action']]}")


def interactive_test(env, agent):
    """交互式测试,逐步显示agent行为"""
    print("\n" + "=" * 70)
    print("交互式测试模式")
    print("=" * 70)
    print("按Enter键继续下一步,输入'q'退出")
    
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    action_names = ['向上', '不动', '向下']
    
    while not done:
        print(f"\n--- Step {step} ---")
        print(f"当前观察: 球X={obs[0]}, 球Y={obs[1]}, 板子={obs[2]}, 奖励信号={obs[3]}")
        
        # Agent决策
        action = agent.step(obs, learning=False)
        paddle_action = int(action[2]) if len(action) > 2 else 1
        
        print(f"Agent动作: {action_names[paddle_action]}")
        
        user_input = input("按Enter继续 (q退出): ")
        if user_input.lower() == 'q':
            print("退出交互式测试")
            break
        
        obs, reward, done, info = env.step(paddle_action)
        total_reward += reward
        step += 1
        
        if reward != 0:
            print(f">>> 获得奖励: {reward:.2f}")
        
        if done:
            print(f"\n游戏结束! 总奖励: {total_reward:.2f}, 总步数: {step}")


if __name__ == "__main__":
    # 创建环境
    print("创建测试环境...")
    env = PongEnvActiveInference(grid_x=8, grid_y=8, grid_paddle=8)
    
    # 加载模型
    model_path = './models/best_agent.pkl'
    
    if not os.path.exists(model_path):
        print(f"\n错误: 找不到模型文件 {model_path}")
        print("请先运行 train_pong_ai.py 训练模型")
        exit(1)
    
    agent = load_agent(model_path, env)
    
    # 测试agent
    print("\n开始标准测试...")
    test_results = test_agent(
        env, 
        agent, 
        num_episodes=20, 
        render=False, 
        verbose=True
    )
    
    # 打印测试总结
    print_test_summary(test_results)
    
    # 可视化测试结果
    visualize_test_results(test_results, save_dir='./figures')
    
    # 比较成功和失败的轮次
    compare_episodes(test_results)
    
    # 询问是否进行交互式测试
    print("\n" + "=" * 70)
    user_choice = input("是否进行交互式测试? (y/n): ")
    if user_choice.lower() == 'y':
        interactive_test(env, agent)
    
    print("\n测试程序执行完成!")
