import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import random


class ExperimentCondition(Enum):
    """实验条件枚举"""
    Stimulus = 0
    NoFeedback = 1
    Control = 2


class GameEvent(Enum):
    """游戏事件枚举"""
    NONE = 0
    BALL_HIT_PLAYER_PADDLE = 1
    PLAYER_MISSED = 2


class PongEnv(gym.Env):
    """
    Pong游戏强化学习环境
    
    状态空间: (球x坐标, 球y坐标, 板子底部位置, 板子顶部位置)
    动作空间: 0=不动, 1=向上, 2=向下
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, discretize=True, grid_size=20):
        """
        初始化Pong环境
        
        Args:
            render_mode: 渲染模式 ('human', 'rgb_array', None)
            discretize: 是否离散化状态空间
            grid_size: 离散化网格大小 (越大状态空间越小)
        """
        super().__init__()
        
        # 游戏参数
        self.game_width = 640
        self.game_height = 480
        self.paddle_width = 10
        self.paddle_height = 160
        self.ball_size = 10
        self.paddle_speed = 20.0
        
        # 离散化参数
        self.discretize = discretize
        self.grid_size = grid_size
        
        # 动作空间: 0=不动, 1=向上, 2=向下
        self.action_space = spaces.Discrete(3)
        
        # 状态空间设计
        if discretize:
            # 离散化状态空间
            # 计算离散化后的维度
            self.x_bins = self.game_width // grid_size  # 例如 640/20 = 32
            self.y_bins = self.game_height // grid_size  # 例如 480/20 = 24
            self.paddle_bins = self.game_height // grid_size  # 板子位置的离散化
            
            # 状态空间: (球x, 球y, 板子底部, 板子顶部)
            # 总状态数约为: 32 * 24 * 24 * 24 = 442,368 (grid_size=20时)
            self.observation_space = spaces.MultiDiscrete([
                self.x_bins,      # 球x坐标 (0 to x_bins-1)
                self.y_bins,      # 球y坐标 (0 to y_bins-1)
                self.paddle_bins, # 板子底部 (0 to paddle_bins-1)
                self.paddle_bins  # 板子顶部 (0 to paddle_bins-1)
            ])
            
            print(f"离散化状态空间:")
            print(f"  球X维度: {self.x_bins}")
            print(f"  球Y维度: {self.y_bins}")
            print(f"  板子位置维度: {self.paddle_bins}")
            print(f"  总状态数约: {self.x_bins * self.y_bins * self.paddle_bins * self.paddle_bins:,}")
        else:
            # 连续状态空间
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0], dtype=np.float32),
                high=np.array([
                    self.game_width,
                    self.game_height,
                    self.game_height,
                    self.game_height
                ], dtype=np.float32),
                dtype=np.float32
            )
        
        # 游戏状态变量
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_speed_x = 0.0
        self.ball_speed_y = 0.0
        self.paddle_y = 0.0
        self.bounces_in_rally = 0
        self.current_condition = ExperimentCondition.Stimulus
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
    def _get_obs(self):
        """获取当前状态观察"""
        if self.discretize:
            # 离散化状态
            ball_x_discrete = int(self.ball_x / self.grid_size)
            ball_y_discrete = int(self.ball_y / self.grid_size)
            paddle_bottom_discrete = int(self.paddle_y / self.grid_size)
            paddle_top_discrete = int((self.paddle_y + self.paddle_height) / self.grid_size)
            
            # 确保在合法范围内
            ball_x_discrete = np.clip(ball_x_discrete, 0, self.x_bins - 1)
            ball_y_discrete = np.clip(ball_y_discrete, 0, self.y_bins - 1)
            paddle_bottom_discrete = np.clip(paddle_bottom_discrete, 0, self.paddle_bins - 1)
            paddle_top_discrete = np.clip(paddle_top_discrete, 0, self.paddle_bins - 1)
            
            return np.array([
                ball_x_discrete,
                ball_y_discrete,
                paddle_bottom_discrete,
                paddle_top_discrete
            ], dtype=np.int64)
        else:
            # 连续状态
            return np.array([
                self.ball_x,
                self.ball_y,
                self.paddle_y,
                self.paddle_y + self.paddle_height
            ], dtype=np.float32)
    
    def _get_info(self):
        """获取额外信息"""
        return {
            'bounces': self.bounces_in_rally,
            'condition': self.current_condition.name,
            'ball_speed': (self.ball_speed_x, self.ball_speed_y)
        }
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 重置球的位置和速度
        self._reset_ball(random_vector=True)
        
        # 重置板子位置
        self.paddle_y = self.game_height / 2 - self.paddle_height / 2
        
        # 重置计数器
        self.bounces_in_rally = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _reset_ball(self, random_vector=True):
        """重置球的位置和速度"""
        self.ball_x = float(self.game_width)
        self.ball_y = float(self.game_height) / 2.0
        
        if random_vector:
            self.ball_speed_x = -4.0 if random.randint(0, 1) == 0 else -3.0
            speed_y = random.randint(1, 3)
            if random.randint(0, 1) == 0:
                speed_y = -speed_y
            self.ball_speed_y = float(speed_y)
        else:
            self.ball_speed_x = -3.5
            self.ball_speed_y = 1.5
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 0=不动, 1=向上, 2=向下
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 根据动作移动板子
        paddle_movement = 0
        if action == 1:  # 向上
            paddle_movement = -1
        elif action == 2:  # 向下
            paddle_movement = 1
        
        # 更新板子位置
        self.paddle_y += paddle_movement * self.paddle_speed
        
        # 限制板子在屏幕内
        self.paddle_y = np.clip(self.paddle_y, 0, self.game_height - self.paddle_height)
        
        # 更新球的位置
        event = self._update_ball_position()
        
        # 计算奖励
        reward = 0.0
        terminated = False
        
        if event == GameEvent.BALL_HIT_PLAYER_PADDLE:
            reward = 1.0  # 成功接到球
        elif event == GameEvent.PLAYER_MISSED:
            reward = -1.0  # 未接到球
            terminated = True  # 一局结束
        else:
            reward = 0.0  # 普通状态，小惩罚鼓励快速结束
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _update_ball_position(self):
        """更新球的位置并检测碰撞"""
        new_ball_x = self.ball_x + self.ball_speed_x
        new_ball_y = self.ball_y + self.ball_speed_y
        
        event = GameEvent.NONE
        
        # X轴碰撞检测
        if self.ball_speed_x < 0.0 and self._check_paddle_collision(new_ball_x, new_ball_y):
            # 球拍碰撞
            self.ball_speed_x = -self.ball_speed_x
            self.ball_x = -new_ball_x
            self.ball_y = new_ball_y
            event = GameEvent.BALL_HIT_PLAYER_PADDLE
        else:
            # 检查左右边界
            if new_ball_x <= 0.0:
                # 球越过左侧边界（玩家未接住）
                self.ball_x = 0.0
                self.ball_speed_x = -self.ball_speed_x
                
                if self.current_condition != ExperimentCondition.NoFeedback:
                    self._reset_ball(random_vector=True)
                
                event = GameEvent.PLAYER_MISSED
            elif new_ball_x >= self.game_width:
                # 球碰到右侧边界
                self.ball_x = float(self.game_width)
                self.ball_speed_x = -self.ball_speed_x
            else:
                # 正常移动
                self.ball_x = new_ball_x
        
        # Y轴边界碰撞
        if new_ball_y <= 0.0:
            self.ball_y = 0.0
            self.ball_speed_y = -self.ball_speed_y
        elif new_ball_y >= self.game_height:
            self.ball_y = float(self.game_height)
            self.ball_speed_y = -self.ball_speed_y
        else:
            if event != GameEvent.BALL_HIT_PLAYER_PADDLE:  # 避免重复更新
                self.ball_y = new_ball_y
        
        return event
    
    def _check_paddle_collision(self, new_ball_x, new_ball_y):
        """检查球拍碰撞"""
        if new_ball_x > 0.0:
            return False
        
        # 计算球轨迹与x=0的交点Y坐标
        if new_ball_x != self.ball_x:
            y_intersect = self.ball_y + (0.0 - self.ball_x) * (new_ball_y - self.ball_y) / (new_ball_x - self.ball_x)
        else:
            y_intersect = new_ball_y
        
        # 检查交点是否在球拍范围内
        if self.paddle_y <= y_intersect <= self.paddle_y + self.paddle_height:
            self.bounces_in_rally += 1
            return True
        
        return False
    
    def render(self):
        """渲染游戏画面"""
        if self.render_mode is None:
            return
        
        try:
            import pygame
        except ImportError:
            raise ImportError("需要安装pygame来渲染游戏画面: pip install pygame")
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.game_width, self.game_height))
                pygame.display.set_caption("Pong RL Environment")
            else:
                self.screen = pygame.Surface((self.game_width, self.game_height))
            self.clock = pygame.time.Clock()
        
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 绘制球拍
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (0, int(self.paddle_y), self.paddle_width, self.paddle_height)
        )
        
        # 绘制球
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(self.ball_x), int(self.ball_y)),
            self.ball_size // 2
        )
        
        # 绘制中线
        for i in range(0, self.game_height, 20):
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (self.game_width // 2 - 2, i, 4, 10)
            )
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None


# 使用示例
if __name__ == "__main__":
    # 创建环境（默认离散化，grid_size=20）
    env = PongEnv(render_mode=None, discretize=True, grid_size=20)
    
    print("=" * 60)
    print("Pong强化学习环境")
    print("=" * 60)
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print()
    
    # 测试环境
    obs, info = env.reset()
    print(f"初始状态: {obs}")
    print(f"初始信息: {info}")
    print()
    
    # 运行几步
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode结束于第{i+1}步")
            print(f"总奖励: {total_reward}")
            print(f"弹跳次数: {info['bounces']}")
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("不同grid_size的状态空间对比:")
    print("=" * 60)
    for grid_size in [10, 20, 40, 80]:
        env_test = PongEnv(discretize=True, grid_size=grid_size)
        x_bins = env_test.x_bins
        y_bins = env_test.y_bins
        paddle_bins = env_test.paddle_bins
        total_states = x_bins * y_bins * paddle_bins * paddle_bins
        print(f"grid_size={grid_size:2d}: {x_bins:2d}x{y_bins:2d}x{paddle_bins:2d}x{paddle_bins:2d} = {total_states:,} 状态")
        env_test.close()