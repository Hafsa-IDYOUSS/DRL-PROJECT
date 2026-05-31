import json
from collections import Counter, defaultdict

from stable_baselines3 import PPO

from rl_environment import MedicalImageEnv


env = MedicalImageEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=2000
)

model.save(
    "../models/ppo_augmentation"
)

sequence_counter = Counter()
sequence_rewards = defaultdict(list)

for episode in range(150):

    obs, _ = env.reset()

    sequence = []
    final_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        action, _ = model.predict(obs)
        action = int(action)

        sequence.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            final_reward = float(reward)

    sequence_key = "-".join(str(a) for a in sequence)

    sequence_counter[sequence_key] += 1
    sequence_rewards[sequence_key].append(final_reward)

average_sequence_rewards = {}

for sequence_key, rewards in sequence_rewards.items():
    average_sequence_rewards[sequence_key] = sum(rewards) / len(rewards)

best_sequence = max(
    average_sequence_rewards,
    key=average_sequence_rewards.get
)

best_sequence_actions = [
    int(x) for x in best_sequence.split("-")
]

policy = {
    "best_sequence": best_sequence,
    "best_sequence_actions": best_sequence_actions,
    "best_average_reward": average_sequence_rewards[best_sequence],
    "average_sequence_rewards": average_sequence_rewards,
    "sequence_counts": dict(sequence_counter),
    "actions": {
        "0": "STOP",
        "1": "contrast_plus",
        "2": "contrast_minus",
        "3": "brightness_plus",
        "4": "brightness_minus",
        "5": "CLAHE",
        "6": "denoise",
        "7": "sharpen",
        "8": "gamma_correction"
    },
    "reward_formula": "R = (P_final(y) - P_initial(y)) - lambda * N",
    "lambda": 0.01,
    "max_actions": 4
}

with open("../results/rl_policy.json", "w") as f:
    json.dump(policy, f, indent=4)

print("DRL policy trained.")
print("Best sequence:", best_sequence)
print("Best average reward:", average_sequence_rewards[best_sequence])
print("Sequence counts:", dict(sequence_counter))