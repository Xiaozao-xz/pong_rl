#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主动推理乒乓球游戏环境与反事实学习Agent训练
基于pymdp框架的主动推理实现
"""

import numpy as np
import matplotlib.pyplot as plt
from pymdp import utils
from agent_cl import cl_agent
import time

class PongEnvActiveInference:
    """
    适配主动推理的乒乓球游戏环境
    
    状态因子:
    - 球的X位置 (离散化)
    - 球的Y位置 (离散化)
    - 板子Y位置 (离散化)
    
    观察模态:
    - 球位置观察 (X, Y)
    - 板子位置观察
    - 奖励信号 (成功/失败/中性)
    
    动作:
    - 0: 不动
    - 1: 向上
    - 2: 向下
    """
    
    def __init__(self, grid_x=8, grid_y=8, grid_paddle=8):
        """
        初始化环境
        
        Args:
            grid_x: X轴离散化网格数
            grid_y: Y轴离散化网格数
            grid_paddle: 板子位置离散化网格数
        """
        # 游戏参数
        self.game_width = 640
        self.game_height = 480
        self.paddle_width = 10
        self.paddle_height = 160
        self.ball_size = 10
        self.paddle_speed = 20.0
        
        # 离散化参数
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_paddle = grid_paddle
        
        # 状态空间维度
        self.num_states = [grid_x, grid_y, grid_paddle]  # [ball_x, ball_y, paddle_y]
        self.num_factors = len(self.num_states)
        
        # 观察空间维度
        self.num_obs = [
            grid_x,           # 观察模态0: 球X位置
            grid_y,           # 观察模态1: 球Y位置
            grid_paddle,      # 观察模态2: 板子位置
            3                 # 观察模态3: 奖励信号 (失败/中性/成功)
        ]
        self.num_modalities = len(self.num_obs)
        
        # 控制空间
        self.num_controls = [1, 1, 3]  # 只有板子位置是可控的
        
        # 游戏状态
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_speed_x = 0.0
        self.ball_speed_y = 0.0
        self.paddle_y = 0.0
        
        self.step_count = 0
        self.max_steps = 200
        self.total_reward = 0
        
        # 构建生成模型
        self._build_generative_model()
        
    def _build_generative_model(self):
        """构建主动推理的生成模型 (A, B, C, D)"""
        
        # A矩阵: 观察模型 P(o|s)
        self.A = utils.obj_array(self.num_modalities)
        
        # A[0]: 球X位置观察 - 确定性观察
        self.A[0] = np.zeros([self.num_obs[0]] + self.num_states)
        for x in range(self.grid_x):
            self.A[0][x, x, :, :] = 1.0
            
        # A[1]: 球Y位置观察 - 确定性观察
        self.A[1] = np.zeros([self.num_obs[1]] + self.num_states)
        for y in range(self.grid_y):
            self.A[1][y, :, y, :] = 1.0
            
        # A[2]: 板子位置观察 - 确定性观察
        self.A[2] = np.zeros([self.num_obs[2]] + self.num_states)
        for p in range(self.grid_paddle):
            self.A[2][p, :, :, p] = 1.0
            
        # A[3]: 奖励信号观察 - 基于球和板子的相对位置
        self.A[3] = np.zeros([self.num_obs[3]] + self.num_states)
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                for p in range(self.grid_paddle):
                    # 失败: 球到达最左侧且没有接到
                    if x == 0:
                        paddle_top = int(p * self.game_height / self.grid_paddle)
                        paddle_bottom = paddle_top + self.paddle_height
                        ball_y_pos = int(y * self.game_height / self.grid_y)
                        
                        if paddle_top <= ball_y_pos <= paddle_bottom:
                            self.A[3][2, x, y, p] = 1.0  # 成功
                        else:
                            self.A[3][0, x, y, p] = 1.0  # 失败
                    else:
                        self.A[3][1, x, y, p] = 1.0  # 中性
        
        # B矩阵: 转移模型 P(s'|s,a)
        self.B = utils.obj_array(self.num_factors)
        
        # B[0]: 球X位置转移 - 不可控,向左移动
        self.B[0] = np.zeros([self.grid_x, self.grid_x, 1])
        for x in range(self.grid_x):
            if x > 0:
                self.B[0][x-1, x, 0] = 0.9  # 向左移动
                self.B[0][x, x, 0] = 0.1    # 保持不变
            else:
                self.B[0][x, x, 0] = 1.0    # 边界反弹
        
        # B[1]: 球Y位置转移 - 不可控,随机移动
        self.B[1] = np.zeros([self.grid_y, self.grid_y, 1])
        for y in range(self.grid_y):
            if y > 0:
                self.B[1][y-1, y, 0] = 0.3  # 向上
            self.B[1][y, y, 0] = 0.4        # 保持
            if y < self.grid_y - 1:
                self.B[1][y+1, y, 0] = 0.3  # 向下
            # 归一化
            self.B[1][:, y, 0] /= self.B[1][:, y, 0].sum()
        
        # B[2]: 板子位置转移 - 可控
        self.B[2] = np.zeros([self.grid_paddle, self.grid_paddle, 3])
        for p in range(self.grid_paddle):
            # 动作0: 向上
            if p > 0:
                self.B[2][p-1, p, 0] = 1.0
            else:
                self.B[2][p, p, 0] = 1.0
            
            # 动作1: 不动
            self.B[2][p, p, 1] = 1.0
            
            # 动作2: 向下
            if p < self.grid_paddle - 1:
                self.B[2][p+1, p, 2] = 1.0
            else:
                self.B[2][p, p, 2] = 1.0
        
        # C矩阵: 偏好 (prior preferences)
        self.C = utils.obj_array_zeros(self.num_obs)
        # 只对奖励信号有偏好
        self.C[3][0] = -3.0  # 强烈避免失败
        self.C[3][1] = 0.0   # 中性状态
        self.C[3][2] = 3.0   # 强烈偏好成功
        
        # D矩阵: 初始状态先验
        self.D = utils.obj_array_uniform(self.num_states)
        
    def reset(self):
        """重置环境"""
        # 重置球位置和速度
        self.ball_x = float(self.game_width * 0.8)
        self.ball_y = float(self.game_height / 2)
        self.ball_speed_x = -3.5
        self.ball_speed_y = np.random.choice([-2.0, -1.0, 1.0, 2.0])
        
        # 重置板子位置
        self.paddle_y = self.game_height / 2 - self.paddle_height / 2
        
        self.step_count = 0
        self.total_reward = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """获取当前观察"""
        # 离散化连续状态
        ball_x_discrete = int(np.clip(self.ball_x / self.game_width * self.grid_x, 0, self.grid_x - 1))
        ball_y_discrete = int(np.clip(self.ball_y / self.game_height * self.grid_y, 0, self.grid_y - 1))
        paddle_discrete = int(np.clip(self.paddle_y / self.game_height * self.grid_paddle, 0, self.grid_paddle - 1))
        
        # 计算奖励信号
        reward_signal = 1  # 中性
        if ball_x_discrete == 0:
            # 检查是否接到球
            paddle_top = self.paddle_y
            paddle_bottom = self.paddle_y + self.paddle_height
            if paddle_top <= self.ball_y <= paddle_bottom:
                reward_signal = 2  # 成功
            else:
                reward_signal = 0  # 失败
        
        # 返回观察元组
        return (ball_x_discrete, ball_y_discrete, paddle_discrete, reward_signal)
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 控制动作 [0=向上, 1=不动, 2=向下]
        
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # 执行板子移动
        if action == 0:  # 向上
            self.paddle_y -= self.paddle_speed
        elif action == 2:  # 向下
            self.paddle_y += self.paddle_speed
        
        # 限制板子位置
        self.paddle_y = np.clip(self.paddle_y, 0, self.game_height - self.paddle_height)
        
        # 更新球位置
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        
        # Y轴边界碰撞
        if self.ball_y <= 0 or self.ball_y >= self.game_height:
            self.ball_speed_y = -self.ball_speed_y
            self.ball_y = np.clip(self.ball_y, 0, self.game_height)
        
        # X轴碰撞检测
        reward = 0
        done = False
        
        if self.ball_x <= 0:
            # 检查是否接到球
            if self.paddle_y <= self.ball_y <= self.paddle_y + self.paddle_height:
                # 成功接到
                reward = 1.0
                self.ball_speed_x = -self.ball_speed_x
                self.ball_x = 0
            else:
                # 失败
                reward = -1.0
                done = True
        elif self.ball_x >= self.game_width:
            # 右侧反弹
            self.ball_speed_x = -self.ball_speed_x
            self.ball_x = self.game_width
        
        # 超过最大步数
        if self.step_count >= self.max_steps:
            done = True
        
        self.total_reward += reward
        
        obs = self._get_obs()
        info = {'total_reward': self.total_reward, 'steps': self.step_count}
        
        return obs, reward, done, info


def train_cl_agent(env, num_episodes=100, save_history=True):
    """
    训练反事实学习agent
    
    Args:
        env: 游戏环境
        num_episodes: 训练轮数
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
        'success_rate': []
    }
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            # Agent观察环境并决策
            action = agent.step(obs, learning=True)
            
            # 映射动作 (agent输出的是因子动作)
            paddle_action = int(action[2]) if len(action) > 2 else 1
            
            # 环境步进
            obs, reward, done, info = env.step(paddle_action)
            episode_reward += reward
            episode_steps += 1
            
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
            
            if episode_reward > 0:
                success_count += 1
            
            if (episode + 1) % 10 == 0:
                recent_success = success_count / 10
                history['success_rate'].append(recent_success)
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Steps: {episode_steps} | "
                      f"Success Rate: {recent_success:.2%}")
                success_count = 0
    
    print("\n训练完成!")
    print("=" * 70)
    
    return agent, history


def visualize_training(history):
    """可视化训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 奖励曲线
    axes[0].plot(history['rewards'], alpha=0.6)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # 步数曲线
    axes[1].plot(history['steps'], alpha=0.6, color='orange')
    axes[1].set_title('Episode Steps')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True, alpha=0.3)
    
    # 成功率曲线
    if len(history['success_rate']) > 0:
        x = np.arange(10, len(history['rewards']) + 1, 10)
        axes[2].plot(x, history['success_rate'], marker='o', color='green')
        axes[2].set_title('Success Rate (per 10 episodes)')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Success Rate')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("训练历史图已保存为 training_history.png")
    plt.show()


def test_agent(env, agent, num_episodes=10, render=False):
    """
    测试训练好的agent
    
    Args:
        env: 环境
        agent: 训练好的agent
        num_episodes: 测试轮数
        render: 是否显示游戏画面
    """
    print("\n" + "=" * 70)
    print("开始测试Agent")
    print("=" * 70)
    
    test_rewards = []
    test_steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            # Agent决策 (不学习)
            action = agent.step(obs, learning=False)
            paddle_action = int(action[2]) if len(action) > 2 else 1
            
            obs, reward, done, info = env.step(paddle_action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                time.sleep(0.05)  # 放慢显示
        
        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)
        
        if episode_reward > 0:
            success_count += 1
        
        print(f"Test Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Steps: {episode_steps}")
    
    # 统计结果
    avg_reward = np.mean(test_rewards)
    avg_steps = np.mean(test_steps)
    success_rate = success_count / num_episodes
    
    print("\n" + "=" * 70)
    print("测试结果:")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均步数: {avg_steps:.1f}")
    print(f"  成功率: {success_rate:.2%}")
    print("=" * 70)
    
    return test_rewards, test_steps, success_rate


if __name__ == "__main__":
    # 创建环境
    env = PongEnvActiveInference(grid_x=8, grid_y=8, grid_paddle=8)
    
    # 训练agent
    agent, history = train_cl_agent(env, num_episodes=100, save_history=True)
    
    # 可视化训练过程
    visualize_training(history)
    
    # 测试agent
    test_agent(env, agent, num_episodes=20, render=False)
    
    print("\n程序执行完成!")
