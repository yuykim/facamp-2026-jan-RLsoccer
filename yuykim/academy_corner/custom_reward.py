import numpy as np
import gym

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cb_indices = [2, 3]
        self.kicker_index = -1
        self.is_cross_active = False  # 크로스가 공중에 떠 있는 상태인가?
        self.last_cross_time = 0      # 크로스 시점 기록 (타이밍 체크용)

    def reset(self, **kwargs):
        self.kicker_index = -1
        self.is_cross_active = False
        self.last_cross_time = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self.obs_to_dict(obs)
        
        ball_pos = obs_dict["ball"][:2]
        active_idx = np.argmax(obs_dict["active_player"])
        owned_team = np.argmax(obs_dict["ball_ownership"]) # 0:없음, 1:아군, 2:적군

        # --- [1. 크로스 시도 감지] ---
        # 키커가 롱킥(10)을 시도하면 '크로스 모드' 활성화
        if action == 10 and owned_team == 1:
            if ball_pos[0] > 0.7: # 코너 부근에서 찼을 때
                self.is_cross_active = True
                self.kicker_index = active_idx
                reward += 0.5 # 크로스 시도 자체에 대한 칭찬
                print(f">> 크로스 올라감! (Active)")

        # --- [2. 핵심: 원터치 슈팅 연계 보상] ---
        if self.is_cross_active:
            # 공이 박스 안에 들어왔고, 아군이 다시 소유했을 때
            if ball_pos[0] > 0.75 and abs(ball_pos[1]) < 0.25:
                if owned_team == 1 and active_idx != self.kicker_index:
                    
                    # [전술 A] 센터백이 공을 받자마자 슛(12)을 시도할 때 (초대박 보상)
                    if active_idx in self.cb_indices:
                        if action == 12:
                            reward += 5.0  # 원터치 슈팅에 강력한 보상
                            print(f">> ★ 센터백({active_idx}) 원터치 헤더/발리 슈팅!! (+5.0)")
                        else:
                            reward += 0.5  # 공을 받기만 해도 칭찬
                    
                    # 공을 받았으므로 크로스 시퀀스 종료
                    self.is_cross_active = False

        # --- [3. 골키퍼 방해 보상 (상시)] ---
        if ball_pos[0] > 0.7:
            opp_gk_pos = obs_dict["right_team"][0:2]
            for i in range(1, 11):
                if i in self.cb_indices or i == self.kicker_index: continue
                p_pos = obs_dict["left_team"][i*2 : i*2+2]
                if np.linalg.norm(p_pos - opp_gk_pos) < 0.05:
                    reward += 0.02

        # --- [4. 결과 처리] ---
        # 공이 뺏기면 리셋 (효율성)
        if owned_team == 2:
            done = True
            # 감점 제거 (사용자 요청: 나가는 것에 대한 감점 없음)
        
        # 득점 보상
        if reward > 0.9: # 슛 시도 후 골이 들어가면
            reward += 10.0
            print(">> GOAL!!!")

        return obs, reward, done, info

    def obs_to_dict(self, obs):
        # (기존 obs_to_dict 로직 유지)
        obs_dict = {}
        obs_dict["left_team"] = obs[0:22]
        obs_dict["right_team"] = obs[44:66]
        obs_dict["ball"] = obs[88:91]
        obs_dict["ball_ownership"] = obs[94:97]
        obs_dict["active_player"] = obs[97:108]
        return obs_dict