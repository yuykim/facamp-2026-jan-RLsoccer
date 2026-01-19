import gym
import numpy as np
from gfootball.env import football_action_set

class RuleBasedAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 상태 저장을 위한 변수 추가
        self.is_aiming = False 
        self.last_target_direction = None

    def obs_to_dict(self, obs):
        obs_dict = {}
        obs_dict["left_team"] = obs[0:22]
        obs_dict["left_team_direction"] = obs[22:44]
        obs_dict["right_team"] = obs[44:66]
        obs_dict["right_team_direction"] = obs[66:88]
        obs_dict["ball"] = obs[88:91]
        obs_dict["ball_direction"] = obs[91:94]
        obs_dict["ball_ownership"] = obs[94:97] 
        obs_dict["active_player"] = obs[97:108]
        obs_dict["game_mode"] = obs[108:115]
        return obs_dict

    def get_action(self, obs):
        d = self.obs_to_dict(obs)
        
        # 1. 내 정보
        active_idx = np.argmax(d["active_player"])
        my_x = d["left_team"][active_idx * 2]
        my_y = d["left_team"][active_idx * 2 + 1]
        my_pos = np.array([my_x, my_y])
        
        # 2. 공 정보
        ball_x = d["ball"][0]
        ball_y = d["ball"][1]
        is_my_ball = d["ball_ownership"][1] == 1.0
        
        # 3. 상대 팀 정보 (0번 수비수가 보통 키퍼)
        opponents = d["right_team"].reshape(11, 2)
        keeper_pos = opponents[0]
        keeper_x, keeper_y = keeper_pos
        dist_to_keeper = np.linalg.norm(my_pos - keeper_pos)
        
        # 주변 상황 판단
        opponents_in_front = [opt for opt in opponents if opt[0] > my_x and abs(opt[1] - my_y) < 0.1]
        opponents_behind = [opt for opt in opponents if opt[0] < my_x and (my_x - opt[0]) < 0.2]

        # --- 슈팅 시퀀스 제어 (지난 프레임에서 조준했다면 이번엔 발사) ---
        if self.is_aiming:
            self.is_aiming = False
            return football_action_set.action_shot

        # --- 규칙 설계 ---
        if not is_my_ball:
            # 수비 시: 공을 쫓아감
            if ball_x > my_x + 0.01: return football_action_set.action_right
            if ball_x < my_x - 0.01: return football_action_set.action_left
            if ball_y > my_y + 0.01: return football_action_set.action_bottom
            if ball_y < my_y - 0.01: return football_action_set.action_top
            return football_action_set.action_sprint

        else:
            # 공격 시: 내가 공을 가졌을 때
            
            # [수정된 슈팅 로직] 거리 0.6 이상이거나 키퍼가 근접했을 때
            if my_x > 0.6 or dist_to_keeper < 0.365:
                self.is_aiming = True
                # 키퍼가 나보다 아래(y > my_y) 있으면 위쪽 구석으로 조준
                # 키퍼가 나보다 위(y < my_y) 있으면 아래쪽 구석으로 조준
                if keeper_y > my_y:
                    #print(f"Aiming TOP-RIGHT! Keeper at {keeper_y:.2f}")
                    return football_action_set.action_top_right
                else:
                    #print(f"Aiming BOTTOM-RIGHT! Keeper at {keeper_y:.2f}")
                    return football_action_set.action_bottom_right
            
            # [스프린트 로직] 앞이 비었고 뒤에서 쫓아올 때
            if len(opponents_behind) > 0:
                return football_action_set.action_sprint
            
            # [드리블 로직] 중앙으로 좁히며 전진
            if my_y > 0.05: return football_action_set.action_top_right
            if my_y < -0.05: return football_action_set.action_bottom_right
            return football_action_set.action_right