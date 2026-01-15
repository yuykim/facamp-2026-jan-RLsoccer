import gym
import numpy as np

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0

    def reset(self, **kwargs):
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        d = self.obs_to_dict(obs)
        
        # 게임 모드 확인 및 시간 측정
        game_mode = np.argmax(d["game_mode"])
        if game_mode == 0: 
            self.steps_from_kickoff += 1
        else: 
            self.steps_from_kickoff = 0

        current_ball_owned_by_us = (d["ball_ownership"][1] == 1.0)
        current_active_player = np.argmax(d["active_player"])
        current_ball_x = d["ball"][0]
        
        # 1. 득점 보상 및 시간 보너스
        if reward > 0:
            time_bonus = max(0, (3000 - self.steps_from_kickoff) * 0.0005) 
            reward += (1.0 + time_bonus) # 기본 득점 + 보너스
            self.steps_from_kickoff = 0

        # 2. 전진 패스 보상 (공격성 유도)
        if self.prev_ball_ownership is not None:
            if current_ball_owned_by_us and self.prev_ball_ownership[1] == 1.0:
                if current_active_player != self.prev_active_player:
                    if self.prev_ball_x is not None and current_ball_x > self.prev_ball_x:
                        forward_gain = current_ball_x - self.prev_ball_x
                        reward += (0.005 + forward_gain * 0.02)
                    self.prev_ball_x = current_ball_x
            
            # 소유권 상실 시 위치 초기화
            if not current_ball_owned_by_us:
                self.prev_ball_x = None
            elif self.prev_ball_x is None and current_ball_owned_by_us:
                self.prev_ball_x = current_ball_x

        # 3. [핵심 수정] 실점 페널티 제거 및 라인 아웃 방지
        # 실점(reward < 0) 시 추가 페널티를 주지 않음 (기본 환경의 -1만 적용)
        
        # 골이 아닌 이유로 경기가 끝나면(라인 아웃) 페널티
        if done and reward == 0 and self.steps_from_kickoff < 2950:
            reward -= 0.1 

        # 4. 소유 유지 보상 (공 가지고 놀면 점수 줌)
        if current_ball_owned_by_us:
            reward += 0.0002

        self.prev_ball_ownership = d["ball_ownership"]
        self.prev_active_player = current_active_player

        return obs, reward, done, info

    def obs_to_dict(self, obs):
        obs_dict = {
            "left_team": obs[0:22],
            "left_team_direction": obs[22:44],
            "right_team": obs[44:66],
            "right_team_direction": obs[66:88],
            "ball": obs[88:91],
            "ball_direction": obs[91:94],
            "ball_ownership": obs[94:97],
            "active_player": obs[97:108],
            "game_mode": obs[108:115],
        }
        return obs_dict