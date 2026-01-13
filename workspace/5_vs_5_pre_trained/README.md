# Pre-trained model 5_vs_5

- Scenario: 5_vs_5 (소규모 전술 학습에 최적화)
- Representation: simple115v2 (선수 위치, 공의 속도 등을 115개의 벡터로 표현)
- Rewards: scoring,checkpoint (단순히 골을 넣는 것뿐만 아니라, 상대 진영으로 공을 전진시키는 중간 과정에 보상 부여)
- Training Steps: 총 10,000,000 Steps 목표

---

| **Parameter** | **Value** | **Description** |
| --- | --- | --- |
| `n_envs` | 8 | CPU 코어를 활용한 8개 환경 병렬 수집 |
| `n_steps` | 2048 | 업데이트당 각 환경에서 수집하는 데이터 샘플 수 |
| `batch_size` | 64 | 신경망 업데이트 시 사용하는 미니배치 크기 |
| `learning_rate` | 0.0003 | 가중치 업데이트 속도 (기본값) |
| `save_freq` | 500,000 | 50만 스텝마다 중간 체크포인트 자동 저장 |