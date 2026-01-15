import gym
import numpy as np

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0  # 킥오프 이후 흐른 스텝 카운트

    def reset(self, **kwargs):
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        d = self.obs_to_dict(obs)
        
        # 게임 모드 확인 (킥오프나 일반 경기 중일 때 스텝 카운트 증가)
        # game_mode: 0:Normal, 1:KickOff, 2:GoalKick, 3:FreeKick, 4:Corner, 5:ThrowIn, 6:Penalty
        game_mode = np.argmax(d["game_mode"])
        if game_mode == 0: # 정상 경기 진행 중일 때만 시간 흐름 체크
            self.steps_from_kickoff += 1
        elif game_mode == 1: # 킥오프 시점에는 카운트 초기화
            self.steps_from_kickoff = 0

        current_ball_owned_by_us = (d["ball_ownership"][1] == 1.0)
        current_active_player = np.argmax(d["active_player"])
        current_ball_x = d["ball"][0]
        
        # 1. 득점 시 시간 보너스 (Time Bonus)
        # 환경에서 기본으로 주는 reward가 1.0이면 우리가 골을 넣은 상황입니다.
        if reward > 0:
            # 빨리 넣을수록 보너스가 커짐 (최대 3000스텝 기준 역산)
            # 1.0 (기본 득점) + 보너스 리워드
            time_bonus = max(0, (3000 - self.steps_from_kickoff) * 0.0005) 
            reward += time_bonus
            self.steps_from_kickoff = 0 # 득점 후 초기화

        # 2. 전진 패스 보상 (기존 유지)
        if self.prev_ball_ownership is not None:
            if current_ball_owned_by_us and self.prev_ball_ownership[1] == 1.0:
                if current_active_player != self.prev_active_player:
                    if self.prev_ball_x is not None and current_ball_x > self.prev_ball_x:
                        forward_gain = current_ball_x - self.prev_ball_x
                        reward += (0.002 + forward_gain * 0.01)
                    else:
                        reward += 0.0005
                    self.prev_ball_x = current_ball_x
            
            if not current_ball_owned_by_us:
                self.prev_ball_x = None
            elif self.prev_ball_x is None and current_ball_owned_by_us:
                self.prev_ball_x = current_ball_x

        # 3. 수비 페널티 및 4. 실점 페널티 (기존 유지)
        if not current_ball_owned_by_us and current_ball_x < -0.5:
            reward -= 0.001
        
        if reward < 0:
            reward -= 0.5

        # 상태 업데이트
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