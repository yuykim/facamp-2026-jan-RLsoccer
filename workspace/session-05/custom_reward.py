import gym
import numpy as np

class CustomReward(gym.Wrapper):
    """
    Goal 전: reward=0 (과정은 stats로만 기록)
    Goal 순간(env_reward > 0): 기록된 stats 기반으로 terminal bonus를 한 번에 지급

    action_set_v1 index (0-based):
      top_right=4, bottom_right=6, shot=12, sprint=13
    """

    TOP_RIGHT_ID = 4
    BOTTOM_RIGHT_ID = 6
    SHOT_ID = 12
    SPRINT_ID = 13

    def __init__(self, env):
        super().__init__(env)
        self.reset_stats()

    def reset_stats(self):
        self.steps = 0
        self.prev_ball_x = None

        # 조준은 "조준(4/6) -> (다음 프레임) shot(12)" 구조를 따라가기 위해 기록
        self.last_aim = None      # "TOP" or "BOTTOM"
        self.last_aim_step = None

        self.stats = {
            "forward_progress": 0.0,     # 공 x 전진 누적
            "backward_progress": 0.0,    # 공 x 후퇴 누적(백패스/후진)
            "entered_shoot_zone": False, # ball_x>0.7 진입 여부
            "hesitation_steps": 0,       # ball_x>0.7에서 shot 안 한 step 수
            "reached_golden_zone": False,# keeper와 거리(0.30~0.40) 들어갔는지
            "did_aim_correctly": False,  # shot 시점에 직전 aim이 맞았는지
            "open_goal_shot": False,     # keeper가 골대에서 많이 벗어난 상태로 shot 했는지
            "shot_taken": False,
            "shot_step": None,
        }

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.reset_stats()
        return obs

    def step(self, action):
        # ---- SB3(VecEnv) 대응: action unwrap ----
        if isinstance(action, np.ndarray):
            action = int(action.squeeze())
        else:
            action = int(action)

        obs, env_reward, done, info = self.env.step(action)
        d = self.obs_to_dict(obs)

        self.steps += 1
        custom_reward = 0.0  # 골 전에는 항상 0

        # ---- 기본 상태 ----
        active_idx = int(np.argmax(d["active_player"]))
        my_pos = np.array(d["left_team"][active_idx*2 : active_idx*2+2], dtype=np.float32)

        ball_pos = np.array(d["ball"][0:2], dtype=np.float32)  # (x,y)
        ball_x = float(ball_pos[0])

        is_my_ball = (d["ball_ownership"][1] == 1.0)

        # keeper: right_team 0번 (네 규칙 기반 가정과 동일)
        keeper_pos = np.array(d["right_team"][0:2], dtype=np.float32)
        goal_pos = np.array([1.0, 0.0], dtype=np.float32)

        # ---- Goal 전: 기록만 ----
        if is_my_ball:
            # 1) 전진/후진 누적(공 x 기준)
            if self.prev_ball_x is not None:
                dx = ball_x - self.prev_ball_x
                if dx > 0:
                    self.stats["forward_progress"] += dx
                elif dx < 0:
                    self.stats["backward_progress"] += -dx

            # 2) 슛 존 진입 & 머뭇거림
            if ball_x > 0.7:
                self.stats["entered_shoot_zone"] = True
                if action != self.SHOT_ID:
                    self.stats["hesitation_steps"] += 1

            # 3) 골든 존(keeper와 거리 0.30~0.40)
            dist_to_keeper = float(np.linalg.norm(my_pos - keeper_pos))
            if 0.30 <= dist_to_keeper <= 0.40:
                self.stats["reached_golden_zone"] = True

            # 4) 조준 기록(규칙 기반처럼: 먼저 top_right/bottom_right로 조준)
            if action == self.TOP_RIGHT_ID:
                self.last_aim = "TOP"
                self.last_aim_step = self.steps
            elif action == self.BOTTOM_RIGHT_ID:
                self.last_aim = "BOTTOM"
                self.last_aim_step = self.steps

            # 5) 슛 기록 + 조준 성공 판정 + 오픈골 판정
            if action == self.SHOT_ID:
                self.stats["shot_taken"] = True
                if self.stats["shot_step"] is None:
                    self.stats["shot_step"] = self.steps

                # (a) 조준 성공 판정: "shot 직전(1~2스텝 내) 조준"이 있었고, keeper 위치에 맞는 방향이면 성공
                if self.last_aim is not None and self.last_aim_step is not None:
                    if (self.steps - self.last_aim_step) <= 2:
                        keeper_y = float(keeper_pos[1])
                        my_y = float(my_pos[1])
                        # 규칙 기반 로직 그대로:
                        # keeper가 나보다 아래(keeper_y > my_y)면 TOP가 유리
                        # keeper가 나보다 위(keeper_y < my_y)면 BOTTOM이 유리
                        if keeper_y > my_y and self.last_aim == "TOP":
                            self.stats["did_aim_correctly"] = True
                        elif keeper_y < my_y and self.last_aim == "BOTTOM":
                            self.stats["did_aim_correctly"] = True

                # (b) 오픈골 판정: keeper가 골대에서 멀면 빈 공간
                keeper_dist_from_goal = float(np.linalg.norm(keeper_pos - goal_pos))
                if keeper_dist_from_goal > 0.15:
                    self.stats["open_goal_shot"] = True

        # ---- Goal 순간에만 정산 ----
        if env_reward > 0:
            base = 1.0

            # 전진 보너스(상한)
            forward_bonus = min(1.0, self.stats["forward_progress"] * 2.0)

            # 백패스 페널티(상한)
            backpass_penalty = min(1.0, self.stats["backward_progress"] * 3.0)

            # 머뭇거림 페널티(상한)
            hesitation_penalty = min(1.0, self.stats["hesitation_steps"] * 0.02)

            # 조준 보너스(가장 크게)
            aim_bonus = 0.7 if self.stats["did_aim_correctly"] else 0.0

            # 골든존 보너스
            golden_bonus = 0.2 if self.stats["reached_golden_zone"] else 0.0

            # 오픈골 보너스
            open_goal_bonus = 0.5 if self.stats["open_goal_shot"] else 0.0

            # 타임 보너스(빠를수록)
            time_bonus = max(0.0, 1.0 - (self.steps / 500.0))

            custom_reward = (base
                             + forward_bonus + aim_bonus + golden_bonus + open_goal_bonus + time_bonus
                             - backpass_penalty - hesitation_penalty)

            # 디버깅 원하면:
            # print("GOAL!", f"R={custom_reward:.3f}", "steps=", self.steps, "stats=", self.stats)

        self.prev_ball_x = ball_x
        return obs, custom_reward, done, info

    def obs_to_dict(self, obs):
        # 네가 준 매핑 그대로 사용
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
