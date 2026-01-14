import gym
import numpy as np

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_ownership = None
        self.prev_active_player = None

    def reset(self, **kwargs):
        self.prev_ball_ownership = None
        self.prev_active_player = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 1. 관측값 딕셔너리 변환
        d = self.obs_to_dict(obs)
        
        # 2. 패스 보상 (Pass Reward)
        # 조건: 공 소유권이 우리 팀(left)인데, 활성 플레이어 인덱스가 바뀌었다면 패스 성공으로 간주
        current_ball_owned_by_us = (d["ball_ownership"][1] == 1.0)
        current_active_player = np.argmax(d["active_player"])
        
        if self.prev_ball_ownership is not None:
            # 우리 팀이 계속 공을 소유 중인데, 조종하는 선수가 바뀌었다면? -> 패스 전달 완료
            if current_ball_owned_by_us and self.prev_ball_ownership[1] == 1.0:
                if current_active_player != self.prev_active_player:
                    reward += 0.001  # 패스 성공 보상 (수치는 학습 보며 조절)

        # 3. 수비 보상 (Defense Reward)
        # 상대방이 우리 골대(x = -1.0) 쪽으로 다가오는 것을 압박할 때 보상
        ball_x = d["ball"][0]
        if not current_ball_owned_by_us: # 공이 우리 소유가 아닐 때 (수비 상황)
            # 우리 팀 선수들과 공 사이의 거리를 좁히면 보상 (압박 유도)
            # 간단하게는 상대방이 우리 진영(ball_x < 0)에서 공을 가질 때 실점 페널티를 강화
            if ball_x < -0.5:
                reward -= 0.001 # 우리 진영 허용 페널티

        # 4. 실점 페널티 (Conceding Penalty) - 매우 중요
        # 기본 scoring 보상은 우리가 넣으면 +1이지만, 먹히면 -1이 아닐 때가 많음
        if reward < 0: # 실점 시
            reward -= 0.5 # 추가 감점으로 수비 의지 고취

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