import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from itertools import count
import argparse

import pong_env


class DQN(nn.Module):
    """DQN网络（与训练时保持一致）"""
    
    def __init__(self, input_size, outputs):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, outputs)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = DQN(input_size=4, outputs=3).to(device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"  Seed: {checkpoint['seed']}")
    print(f"  Training episodes: {len(checkpoint['episode_rewards'])}")
    
    return model, checkpoint


def test_model(env, model, device, num_episodes=10, render=True, fps=30):
    """测试模型性能"""
    
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_hits': [],
        'episode_bounces': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode + 5000)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        total_reward = 0
        num_hits = 0
        
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        for t in count():
            # 选择动作（贪婪策略）
            with torch.no_grad():
                action = model(state).max(1)[1].item()
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if reward > 0:
                num_hits += 1
            
            # 渲染
            if render:
                env.render()
                time.sleep(1.0 / fps)
            
            # 打印步骤信息
            if (t + 1) % 50 == 0:
                print(f"Step {t+1}: Reward={total_reward:.1f}, Hits={num_hits}, Bounces={info['bounces']}")
            
            # 移动到下一个状态
            if not (terminated or truncated):
                state = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
            else:
                break
        
        # 记录结果
        results['episode_rewards'].append(total_reward)
        results['episode_lengths'].append(t + 1)
        results['episode_hits'].append(num_hits)
        results['episode_bounces'].append(info['bounces'])
        
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Steps: {t + 1}")
        print(f"  Hits: {num_hits}")
        print(f"  Bounces: {info['bounces']}")
    
    # 打印总体统计
    print(f"\n{'='*50}")
    print("Overall Test Results")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"Average Length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}")
    print(f"Average Hits: {np.mean(results['episode_hits']):.2f} ± {np.std(results['episode_hits']):.2f}")
    print(f"Average Bounces: {np.mean(results['episode_bounces']):.2f} ± {np.std(results['episode_bounces']):.2f}")
    
    return results


def compare_models(env, model_paths, device, num_episodes=10):
    """比较多个模型的性能"""
    
    all_results = {}
    
    for path in model_paths:
        print(f"\nTesting model: {path}")
        model, checkpoint = load_model(path, device)
        results = test_model(env, model, device, num_episodes=num_episodes, render=False)
        all_results[path] = results
    
    # 打印比较结果
    print(f"\n{'='*70}")
    print("Model Comparison")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Avg Reward':<15} {'Avg Hits':<15} {'Avg Length':<15}")
    print("-" * 70)
    
    for path, results in all_results.items():
        avg_reward = np.mean(results['episode_rewards'])
        avg_hits = np.mean(results['episode_hits'])
        avg_length = np.mean(results['episode_lengths'])
        print(f"{path:<30} {avg_reward:<15.2f} {avg_hits:<15.2f} {avg_length:<15.1f}")


def interactive_play(env, model, device):
    """交互式测试 - 按任意键继续下一步"""
    
    print("\nInteractive Mode - Press Enter to step through the game")
    print("Press 'q' + Enter to quit")
    
    obs, info = env.reset()
    state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    total_reward = 0
    step = 0
    
    while True:
        # 显示当前状态
        print(f"\n--- Step {step} ---")
        print(f"State: {obs}")
        print(f"Bounces: {info['bounces']}")
        
        # 选择动作
        with torch.no_grad():
            q_values = model(state)
            action = q_values.max(1)[1].item()
        
        action_names = ['Stay', 'Up', 'Down']
        print(f"Q-values: {q_values.squeeze().cpu().numpy()}")
        print(f"Action: {action} ({action_names[action]})")
        
        # 等待用户输入
        user_input = input("Press Enter to continue (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        
        # 渲染
        env.render()
        
        if terminated or truncated:
            print(f"\nEpisode finished!")
            print(f"Total Reward: {total_reward}")
            print(f"Total Steps: {step + 1}")
            print(f"Total Bounces: {info['bounces']}")
            break
        
        state = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        obs = next_obs
        step += 1


def main():
    parser = argparse.ArgumentParser(description='Test Pong DQN Model')
    parser.add_argument('--model', type=str, default='pong_dqn_seed_0.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the game')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for rendering')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple models')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - step through game')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    render_mode = 'human' if args.render else None
    env = pong_env.PongEnv(render_mode=render_mode, discretize=True, grid_size=20)
    
    if args.compare:
        # 比较多个模型
        compare_models(env, args.compare, device, num_episodes=args.episodes)
    elif args.interactive:
        # 交互模式
        model, _ = load_model(args.model, device)
        interactive_play(env, model, device)
    else:
        # 标准测试
        model, checkpoint = load_model(args.model, device)
        test_model(env, model, device, num_episodes=args.episodes, 
                  render=args.render, fps=args.fps)
    
    env.close()


if __name__ == "__main__":
    main()