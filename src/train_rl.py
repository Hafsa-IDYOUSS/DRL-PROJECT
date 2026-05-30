from stable_baselines3 import PPO

from rl_environment import MedicalImageEnv


env = MedicalImageEnv()

model = PPO(

    "MlpPolicy",

    env,

    verbose=1

)

model.learn(

    total_timesteps=10000
)

model.save(

    "../models/ppo_augmentation"
)

print("DRL augmentation policy trained!")