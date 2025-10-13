import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pong_env  # 导入我们创建的环境


# 设置matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ============= 超参数设置 =============
BATCH_SIZE = 32
lr = 1e-3
GAMMA = 0.99
Memory_Size = 10000
seeds_num = 10  # 运行的种子数量
num_episodes = 100  # 每个种子的episode数
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
max_frames = 50000  # 每个种子的最大frame数


class ReplayMemory(object):
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """DQN网络 - 适配离散状态空间"""
    
    def __init__(self, input_size, outputs):
        super(DQN, self).__init__()
        
        # 简单的全连接网络
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


def get_state(env):
    """从环境获取状态并转换为tensor"""
    obs, _ = env._get_obs(), env._get_info()
    # obs是numpy array: [ball_x_discrete, ball_y_discrete, paddle_bottom, paddle_top]
    state = torch.from_numpy(obs).float().unsqueeze(0)  # 添加batch维度
    return state


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # 创建policy网络和target网络
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器和经验回放
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(Memory_Size)
        
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """选择动作（epsilon-greedy策略）"""
        if not training:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], 
                              device=device, dtype=torch.long)
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # 创建mask标记非终止状态
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=device, dtype=torch.bool
        )
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # 计算Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算V(s_{t+1})
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # 计算期望Q值
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # 计算Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_net(self):
        """更新target网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_one_episode(env, agent, seed):
    """训练一个episode"""
    obs, info = env.reset(seed=seed)
    state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    total_reward = 0
    num_hits = 0
    
    for t in count():
        # 选择动作
        action = agent.select_action(state, training=True)
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        
        # 统计击球次数
        if reward > 0:
            num_hits += 1
        
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        
        # 获取下一个状态
        if not (terminated or truncated):
            next_state = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        else:
            next_state = None
        
        # 存储transition
        agent.memory.push(state, action, next_state, reward_tensor)
        
        # 移动到下一个状态
        state = next_state
        
        # 优化模型
        agent.optimize_model()
        
        if terminated or truncated:
            return t + 1, total_reward, num_hits, info['bounces']
    
    return t + 1, total_reward, num_hits, info['bounces']


def test_agent(env, agent, num_episodes=10):
    """测试智能体性能"""
    total_rewards = []
    total_hits = []
    episode_lengths = []
    
    for i in range(num_episodes):
        obs, info = env.reset(seed=i + 10000)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        episode_reward = 0
        num_hits = 0
        
        for t in count():
            # 贪婪策略选择动作
            action = agent.select_action(state, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            episode_reward += reward
            if reward > 0:
                num_hits += 1
            
            if not (terminated or truncated):
                state = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
            else:
                break
        
        total_rewards.append(episode_reward)
        total_hits.append(num_hits)
        episode_lengths.append(t + 1)
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_hits': np.mean(total_hits),
        'avg_length': np.mean(episode_lengths),
        'std_reward': np.std(total_rewards),
        'std_hits': np.std(total_hits)
    }


def plot_training_progress(episode_rewards, episode_lengths, window=10):
    """绘制训练进度"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制奖励
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        ax1.plot(moving_avg, label=f'{window}-Episode Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制episode长度
    ax2.plot(episode_lengths, alpha=0.3, label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg = pd.Series(episode_lengths).rolling(window=window).mean()
        ax2.plot(moving_avg, label=f'{window}-Episode Moving Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    print("Training progress plot saved as 'training_progress.png'")


def main():
    """主训练函数"""
    print("=" * 70)
    print("Pong DQN Training")
    print("=" * 70)
    
    # 创建环境
    env = pong_env.PongEnv(render_mode=None, discretize=True, grid_size=20)
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.nvec.sum()  # 总特征数
    action_size = env.action_space.n
    
    print(f"\nEnvironment Info:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # 存储所有种子的结果
    all_seeds_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_hits': [],
        'test_results': []
    }
    
    # 多种子训练
    for seed in range(seeds_num):
        print(f"\n{'=' * 70}")
        print(f"Training Seed {seed + 1}/{seeds_num}")
        print(f"{'=' * 70}")
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 创建智能体
        agent = DQNAgent(state_size=4, action_size=action_size)
        
        episode_rewards = []
        episode_lengths = []
        episode_hits = []
        frame_count = 0
        
        # 训练episodes
        for i_episode in range(num_episodes):
            steps, reward, hits, bounces = train_one_episode(env, agent, seed + i_episode)
            
            episode_rewards.append(reward)
            episode_lengths.append(steps)
            episode_hits.append(hits)
            frame_count += steps
            
            # 更新target网络
            if i_episode % TARGET_UPDATE == 0:
                agent.update_target_net()
            
            # 打印进度
            if (i_episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_hits = np.mean(episode_hits[-10:])
                print(f"Episode {i_episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Avg Hits: {avg_hits:.1f}")
            
            # 检查是否达到最大frame数
            if frame_count >= max_frames:
                print(f"Reached max frames ({max_frames}), stopping training.")
                break
        
        # 测试智能体
        print("\nTesting agent...")
        test_results = test_agent(env, agent, num_episodes=20)
        print(f"Test Results:")
        print(f"  Average Reward: {test_results['avg_reward']:.2f} ± {test_results['std_reward']:.2f}")
        print(f"  Average Hits: {test_results['avg_hits']:.2f} ± {test_results['std_hits']:.2f}")
        print(f"  Average Length: {test_results['avg_length']:.1f}")
        
        # 保存结果
        all_seeds_results['episode_rewards'].append(episode_rewards)
        all_seeds_results['episode_lengths'].append(episode_lengths)
        all_seeds_results['episode_hits'].append(episode_hits)
        all_seeds_results['test_results'].append(test_results)
        
        # 保存模型
        torch.save({
            'seed': seed,
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }, f'pong_dqn_seed_{seed}.pth')
        print(f"Model saved as 'pong_dqn_seed_{seed}.pth'")
    
    # 汇总所有种子的结果
    print(f"\n{'=' * 70}")
    print("Overall Results Across All Seeds")
    print(f"{'=' * 70}")
    
    all_test_rewards = [r['avg_reward'] for r in all_seeds_results['test_results']]
    all_test_hits = [r['avg_hits'] for r in all_seeds_results['test_results']]
    
    print(f"Average Test Reward: {np.mean(all_test_rewards):.2f} ± {np.std(all_test_rewards):.2f}")
    print(f"Average Test Hits: {np.mean(all_test_hits):.2f} ± {np.std(all_test_hits):.2f}")
    
    # 绘制第一个种子的训练曲线
    if len(all_seeds_results['episode_rewards']) > 0:
        plot_training_progress(
            all_seeds_results['episode_rewards'][0],
            all_seeds_results['episode_lengths'][0]
        )
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'seed': range(seeds_num),
        'avg_reward': all_test_rewards,
        'avg_hits': all_test_hits,
    })
    results_df.to_csv('pong_dqn_results.csv', index=False)
    print("\nResults saved to 'pong_dqn_results.csv'")
    
    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()