import gymnasium as gym

from gymnasium import spaces

import numpy as np

from PIL import Image

import torchvision.transforms as transforms


class MedicalImageEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(224, 224, 3),
            dtype=np.uint8
        )

        self.image = np.zeros(
            (224, 224, 3),
            dtype=np.uint8
        )

        self.transformations = {

            0: transforms.RandomRotation(20),

            1: transforms.RandomHorizontalFlip(p=1.0),

            2: transforms.CenterCrop(180),

            3: transforms.ColorJitter(contrast=0.5)

        }

    def reset(self, seed=None, options=None):

        self.image = np.random.randint(
            0,
            255,
            (224, 224, 3),
            dtype=np.uint8
        )

        return self.image, {}

    def step(self, action):

        pil_image = Image.fromarray(self.image)

        transform = self.transformations[action]

        augmented = transform(pil_image)

        augmented = augmented.resize((224, 224))

        augmented = np.array(augmented)

        self.image = augmented

        # reward placeholder
        reward = np.random.uniform(0.7, 1.0)

        terminated = False
        truncated = False

        info = {}

        return (
            self.image,
            reward,
            terminated,
            truncated,
            info
        )