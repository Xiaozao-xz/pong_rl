#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pong_ai.py
训练主动推理乒乓球Agent
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from pong_env_ai import PongEnvActiveInference
from agent_cl import cl_agent


def train_cl_agent(env, num_episodes=100, save_dir='./models', save_history=True):
    """
    训练反事实学习agent
    
    Args:
        env: 游戏环境
        num_episodes: 训练轮数
        save_dir: 模型保存目录
        save_history: 是否保存训练历史
    
    Returns:
        agent, history
    """
    print("=" * 70)
    print("开始训练反事实学习Agent")
    print("=" * 70)
    print(f"状态空间: {env.num_states}")
    print(f"观察空间: {env.num_obs}")
    print(f"控制空间: {env.num_controls}")
    print(f"训练轮数: {num_episodes}")
    print()
    
    # 创建CL agent
    agent = cl_agent(
        A=env.A,
        B=env.B,
        C=env.C,
        D=env.D,
        action_precision=16.0,
        planning_precision=16.0,
        memory_horizon=3,
        gamma_initial=0.5
    )
    
    # 训练历史
    history = {
        'rewards': [],
        'steps': [],
        'success_rate': [],
        'episode_details': []
    }
    
    success_count = 0
    best_avg_reward = float('-inf')
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_events = []
        
        while not done:
            # Agent观察环境并决策
            action = agent.step(obs, learning=True)
            
            # 映射动作 (agent输出的是因子动作)
            paddle_action = int(action[2]) if len(action) > 2 else 1
            
            # 环境步进
            obs, reward, done, info = env.step(paddle_action)
            episode_reward += reward
            episode_steps += 1
            
            # 记录事件
            episode_events.append({
                'step': episode_steps,
                'action': paddle_action,
                'obs': obs,
                'reward': reward
            })
            
            # 更新gamma (风险参数)
            if reward > 0:
                agent.update_gamma(terminated=False, risk=-0.1)
            elif reward < 0:
                agent.update_gamma(terminated=True, risk=0.2)
        
        # 更新CL策略
        agent.update_CL(episode_steps)
        
        # 记录历史
        if save_history:
            history['rewards'].append(episode_reward)
            history['steps'].append(episode_steps)
            history['episode_details'].append({
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': episode_steps,
                'events': episode_events
            })
            
            if episode_reward > 0:
                success_count += 1
            
            # 每10轮统计一次
            if (episode + 1) % 10 == 0:
                recent_success = success_count / 10
                recent_avg_reward = np.mean(history['rewards'][-10:])
                history['success_rate'].append(recent_success)
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Steps: {episode_steps} | "
                      f"Success Rate: {recent_success:.2%} | "
                      f"Avg Reward: {recent_avg_reward:.2f}")
                
                # 保存最佳模型
                if recent_avg_reward > best_avg_reward:
                    best_avg_reward = recent_avg_reward
                    save_agent(agent, save_dir, 'best_agent.pkl')
                    print(f"  → 保存最佳模型 (平均奖励: {best_avg_reward:.2f})")
                
                success_count = 0
    
    print("\n训练完成!")
    print("=" * 70)
    
    # 保存最终模型和历史
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_agent(agent, save_dir, f'final_agent_{timestamp}.pkl')
    save_training_history(history, save_dir, f'training_history_{timestamp}.pkl')
    
    return agent, history


def save_agent(agent, save_dir, filename):
    """保存训练好的agent"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    # 保存agent的关键参数
    agent_data = {
        'CL': agent.CL,
        'Gamma': agent.Gamma,
        'A': agent.A,
        'B': agent.B,
        'C': agent.C,
        'D': agent.D,
        'num_states': agent.num_states,
        'num_obs': agent.num_obs,
        'num_controls': agent.num_controls,
        'action_precision': agent.alpha,
        'planning_precision': agent.gamma,
        'memory_horizon': agent.memory_horizon,
        'gamma_initial': agent.gamma_initial
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(agent_data, f)
    
    print(f"Agent已保存到: {filepath}")


def save_training_history(history, save_dir, filename):
    """保存训练历史"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"训练历史已保存到: {filepath}")


def visualize_training(history, save_dir='./figures'):
    """可视化训练历史"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(history['rewards'], alpha=0.6, label='Episode Reward')
    if len(history['rewards']) >= 10:
        # 添加移动平均
        window = 10
        moving_avg = np.convolve(history['rewards'], 
                                 np.ones(window)/window, 
                                 mode='valid')
        axes[0, 0].plot(range(window-1, len(history['rewards'])), 
                       moving_avg, 
                       color='red', 
                       linewidth=2, 
                       label=f'{window}-Episode Moving Avg')
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 步数曲线
    axes[0, 1].plot(history['steps'], alpha=0.6, color='orange', label='Episode Steps')
    if len(history['steps']) >= 10:
        window = 10
        moving_avg = np.convolve(history['steps'], 
                                 np.ones(window)/window, 
                                 mode='valid')
        axes[0, 1].plot(range(window-1, len(history['steps'])), 
                       moving_avg, 
                       color='red', 
                       linewidth=2, 
                       label=f'{window}-Episode Moving Avg')
    axes[0, 1].set_title('Episode Steps', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 成功率曲线
    if len(history['success_rate']) > 0:
        x = np.arange(10, len(history['rewards']) + 1, 10)
        axes[1, 0].plot(x, history['success_rate'], marker='o', 
                       color='green', linewidth=2, markersize=6)
        axes[1, 0].set_title('Success Rate (per 10 episodes)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].grid(True, alpha=0.3)
    
    # 奖励分布直方图
    axes[1, 1].hist(history['rewards'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(np.mean(history['rewards']), 
                      color='red', 
                      linestyle='--', 
                      linewidth=2, 
                      label=f'Mean: {np.mean(history["rewards"]):.2f}')
    axes[1, 1].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f'training_curves_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {filepath}")
    
    plt.show()


def print_training_summary(history):
    """打印训练总结"""
    print("\n" + "=" * 70)
    print("训练总结")
    print("=" * 70)
    
    rewards = np.array(history['rewards'])
    steps = np.array(history['steps'])
    
    print(f"总训练轮数: {len(rewards)}")
    print(f"\n奖励统计:")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  最大奖励: {np.max(rewards):.2f}")
    print(f"  最小奖励: {np.min(rewards):.2f}")
    print(f"  标准差: {np.std(rewards):.2f}")
    
    print(f"\n步数统计:")
    print(f"  平均步数: {np.mean(steps):.1f}")
    print(f"  最大步数: {np.max(steps)}")
    print(f"  最小步数: {np.min(steps)}")
    
    # 最后10轮表现
    if len(rewards) >= 10:
        last_10_rewards = rewards[-10:]
        last_10_success = np.sum(last_10_rewards > 0)
        print(f"\n最后10轮表现:")
        print(f"  平均奖励: {np.mean(last_10_rewards):.2f}")
        print(f"  成功次数: {last_10_success}/10 ({last_10_success*10}%)")
    
    print("=" * 70)


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 创建环境
    print("创建主动推理乒乓球环境...")
    env = PongEnvActiveInference(grid_x=8, grid_y=8, grid_paddle=8)
    
    # 训练agent
    agent, history = train_cl_agent(
        env, 
        num_episodes=100,
        save_dir='./models',
        save_history=True
    )
    
    # 打印训练总结
    print_training_summary(history)
    
    # 可视化训练过程
    visualize_training(history, save_dir='./figures')
    
    print("\n训练程序执行完成!")
    print("请运行 test_pong_ai.py 来测试训练好的agent")
