import gym
import numpy as np

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0
        self.possession_steps = 0  # 공 소유 시간 카운터 초기화

    def reset(self, **kwargs):
        self.prev_ball_ownership = None
        self.prev_active_player = None
        self.prev_ball_x = None
        self.steps_from_kickoff = 0
        self.possession_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. 먼저 환경의 step을 실행하여 obs와 기본 reward를 가져옴
        obs, reward, done, info = self.env.step(action)
        d = self.obs_to_dict(obs)
        
        # 2. 필요한 데이터 추출 (변수 선언을 로직보다 위로 올림)
        active_idx = np.argmax(d["active_player"])
        my_pos = d["left_team"][active_idx*2 : active_idx*2+2] # 현재 활성 선수 [x, y]
        current_ball_owned_by_us = (d["ball_ownership"][1] == 1.0) # 우리 팀 소유 여부
        current_active_player = active_idx
        current_ball_x = d["ball"][0]
        
        # 3. [추가] 공 소유 정체 방지 로직 (멍 때리기 방지)
        # 같은 선수가 공을 계속 잡고 있는지 체크
        if current_ball_owned_by_us and (self.prev_active_player is not None and current_active_player == self.prev_active_player):
            self.possession_steps += 1
        else:
            self.possession_steps = 0

        # 한 선수가 50스텝(약 5초) 이상 공을 끌면 페널티 부여
        if self.possession_steps > 50:
            reward -= 0.05  # 페널티를 조금 더 강화함
            
        # 4. 슈팅 거리 제한 및 보상
        if action == 12: # Shot Action
            if my_pos[0] < 0.4: # 우리 진영이나 하프라인 근처에서 슛 쏘면 감점
                reward -= 0.1  
            else:
                # 골대(1.0, 0.0)와 가까울수록 추가 보너스
                dist_to_goal = np.linalg.norm(my_pos - np.array([1.0, 0.0]))
                shot_bonus = max(0, 0.1 * (1.0 - dist_to_goal))
                reward += shot_bonus

        # 5. 압박 상황 패스 유도
        if current_ball_owned_by_us:
            opponents = d["right_team"].reshape(11, 2)
            dist_to_opps = np.linalg.norm(opponents - my_pos, axis=1)
            close_opponents = np.sum(dist_to_opps < 0.15)
            
            if close_opponents >= 1: # 주변에 적이 있을 때
                if action in [9, 10, 11]: # 패스 시도 시 보상
                    reward += 0.02 
                elif action == 12: # 억지 슛 시 감점
                    reward -= 0.05

        # 6. 시간 및 게임 모드 관련 로직
        game_mode = np.argmax(d["game_mode"])
        if game_mode == 0: # Normal play
            self.steps_from_kickoff += 1
        else: 
            self.steps_from_kickoff = 0
        
        # 7. 득점 보상 및 시간 보너스
        if reward > 0: # 엔진 기본 보상이 득점(+1)일 때
            time_bonus = max(0, (3000 - self.steps_from_kickoff) * 0.0005) 
            reward += (1.0 + time_bonus)
            self.steps_from_kickoff = 0

        # 8. 전진 패스 보상 (공격성 유도)
        if self.prev_ball_ownership is not None:
            # 우리 팀 소유 유지 중이고 패스가 일어났을 때
            if current_ball_owned_by_us and self.prev_ball_ownership[1] == 1.0:
                if current_active_player != self.prev_active_player:
                    # 공의 위치가 이전보다 전진했는지 확인
                    if self.prev_ball_x is not None and current_ball_x > self.prev_ball_x:
                        forward_gain = current_ball_x - self.prev_ball_x
                        reward += (0.01 + forward_gain * 0.05) # 보상 수치 약간 상향
                    self.prev_ball_x = current_ball_x
            
            # 소유권 상실 시 위치 추적 초기화
            if not current_ball_owned_by_us:
                self.prev_ball_x = None
            elif self.prev_ball_x is None and current_ball_owned_by_us:
                self.prev_ball_x = current_ball_x
    
        # 9. 단순 소유 보상 (매우 작게 유지하여 정체 유발 방지)
        if current_ball_owned_by_us:
            reward += 0.0001

        # 다음 스텝을 위한 상태 저장
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