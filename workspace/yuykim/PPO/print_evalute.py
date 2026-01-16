import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import evalute_PPO
from stable_baselines3 import PPO

# [기존 사용자 함수들: make_env, evaluate_scoring, step_stats, no_concede_stats, global_to_ep_step 포함]

def run_comprehensive_evaluation(model_dir, episodes=10, max_steps=3000):
    model_files = glob.glob(os.path.join(model_dir, "*.zip"))
    model_files.sort()
    
    all_data = []
    env = evalute_PPO.make_env(render=False)

    for path in model_files:
        model_name = os.path.basename(path).replace(".zip", "")
        print(f"~ing: {model_name}...", end=" ", flush=True)
        
        model = PPO.load(path, env=env, device="cpu")
        stats = evalute_PPO.evaluate_scoring(model, env, episodes=episodes, max_steps=max_steps)
        
        # 1. 점수 관련 (평균 및 표준편차)
        avg_home = np.mean(stats["home_score_list"])
        std_home = np.std(stats["home_score_list"])
        avg_away = np.mean(stats["away_score_list"])
        std_away = np.std(stats["away_score_list"])
        
        # 2. 수비 안정성 (무실점 스팬)
        longest_nc, avg_nc, _ = evalute_PPO.no_concede_stats(stats["concede_steps_global"], stats["total_timeline_steps"])
        
        # 3. 골 타이밍 (Fastest Goal)
        h_fast, _, h_avg = evalute_PPO.step_stats(stats["score_steps_global"])

        all_data.append({
            "Model": model_name,
            "Avg_Home": avg_home, "Std_Home": std_home,
            "Avg_Away": avg_away, "Std_Away": std_away,
            "Longest_NC": longest_nc,
            "Avg_NC": avg_nc,
            "Fastest_Goal": h_fast if h_fast is not None else max_steps,
            "Avg_Goal_Step": h_avg if h_avg is not None else max_steps
        })
        print("complete")

    env.close()
    return pd.DataFrame(all_data)

def save_rich_visualization(df, save_name="model_report.jpg"):
    # 3개의 그래프 영역 생성 (점수/분산, 수비 안정성, 골 타이밍)
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4)
    
    x = np.arange(len(df))
    models = df["Model"]

    # --- 그래프 1: 평균 득실점 및 꾸준함 (Error Bar 사용) ---
    axes[0].bar(x - 0.2, df["Avg_Home"], 0.4, yerr=df["Std_Home"], 
                label='Home Score (Avg ± STD)', color='#4C72B0', capsize=5, alpha=0.8)
    axes[0].bar(x + 0.2, df["Avg_Away"], 0.4, yerr=df["Std_Away"], 
                label='Away Score (Avg ± STD)', color='#C44E52', capsize=5, alpha=0.8)
    axes[0].set_title("1. Scoring Performance & Consistency (Lower STD = More Stable)", fontsize=15)
    axes[0].set_ylabel("Goals")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20)
    axes[0].legend()

    # --- 그래프 2: 수비 지표 (최장 vs 평균 무실점 시간) ---
    axes[1].plot(models, df["Longest_NC"], marker='s', color='#55A868', linewidth=3, label="Longest No-Concede")
    axes[1].plot(models, df["Avg_NC"], marker='o', color='#81C784', linestyle='--', label="Avg No-Concede")
    axes[1].set_title("2. Defensive Stability (Span in Steps)", fontsize=15)
    axes[1].set_ylabel("Steps")
    axes[1].set_xticklabels(models, rotation=20)
    axes[1].legend()

    # --- 그래프 3: 공격 효율성 (골 타이밍) ---
    axes[2].scatter(models, df["Fastest_Goal"], color='gold', s=100, edgecolors='black', label="Fastest Goal Step")
    axes[2].plot(models, df["Avg_Goal_Step"], marker='x', color='orange', linestyle=':', label="Avg Goal Step")
    axes[2].set_title("3. Scoring Efficiency (Timing - Lower is Faster)", fontsize=15)
    axes[2].set_ylabel("Steps")
    axes[2].set_xticklabels(models, rotation=20)
    axes[2].invert_yaxis() # 빠른 골이 위로 오도록 Y축 반전
    axes[2].legend()

    # 파일 저장
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\nsave img: {save_name}")
    plt.show()

if __name__ == "__main__":
    MODEL_FOLDER = "./custom_model" # 모델 폴더 경로 확인!
    
    # 1. 전 지표 데이터 수집 (episodes는 분산 측정을 위해 최소 10회 이상 추천)
    report_df = run_comprehensive_evaluation(MODEL_FOLDER, episodes=20, max_steps=3000)
    
    # 2. 파일명 설정 및 저장
    output_file = f"comparison_report_{datetime.now().strftime('%m%d_%H%M')}.jpg"
    save_rich_visualization(report_df, save_name=output_file)