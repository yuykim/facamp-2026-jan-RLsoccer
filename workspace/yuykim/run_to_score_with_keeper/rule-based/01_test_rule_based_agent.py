import gfootball.env as football_env
from stable_baselines3 import PPO
import os
import utils
import rule_based_policy
import time

def main():
    # 1. 환경 생성
    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=True,
        representation="simple115v2",
        write_video=False 
    )

    # 2. 규칙 기반 에이전트 초기화
    agent = rule_based_policy.RuleBasedAgent(env)
    
    t = 0 # 비디오 프레임 번호 (전체 에피소드 통합)
    total_episodes = 10

    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        step_count = 0 # 해당 에피소드의 스텝 카운트
        max_steps = 3000 
        total_reward = 0

        print(f"--- Episode {ep + 1} Start ---")

        # t 대신 step_count를 조건으로 사용해야 모든 에피소드가 정상 작동합니다.
        while (not done) and (step_count < max_steps):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            #time.sleep(0.01)

            frame = env.render(mode="rgb_array")
            if frame is not None:
                utils.save_frame(frame, t) # 프레임 저장은 누적 스텝 t 사용

            if reward != 0:
                print(f"Ep: {ep+1} | Step: {step_count} | Reward: {reward}")

            t += 1
            step_count += 1
            total_reward += reward

        print(f"Episode {ep + 1} Done | Total Reward: {total_reward} | Total Steps: {step_count}")

    env.close()

if __name__ == "__main__":
    utils.cleanup()
    main()
    utils.make_video()