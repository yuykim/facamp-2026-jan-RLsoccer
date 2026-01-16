import gfootball.env as football_env
import rule_based_policy

# 1. 환경 생성
env = football_env.create_environment(
    env_name="academy_run_to_score_with_keeper",
    render=True,
    representation="simple115v2"
)

# 2. 규칙 기반 에이전트 적용
agent = rule_based_policy.RuleBasedAgent(env)

total_episodes = 10

all_reward = 0
all_step = 0

for ep in range(total_episodes):
    obs = agent.reset()  # 경기가 시작될 때 딱 한 번 초기화
    done = False
    step_count = 0
    ep_reward = 0

    print(f"--- ep {ep + 1} start ---")

    while not done:
        # 규칙 기반 액션 결정
        action = agent.get_action(obs)
        
        # 액션 실행 (int로 변환하여 안전하게 전달)
        obs, reward, done, info = agent.step(action)
        
        ep_reward += reward
        step_count += 1

        # 무한 루프 방지를 위한 안전장치 (선택 사항)
        if step_count > 3000:
            break

    print(f"ep {ep + 1} done | total_reawrd: {ep_reward} | total_step: {step_count}")
    all_reward += ep_reward
    all_step += step_count

print("--------------------------------")
print(f"all ep done | all_reawrd: {all_reward} | all_step: {all_step}")

env.close()