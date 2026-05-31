import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from baseline_model import get_model
from rl_reward import final_confidence_reward
from preprocessing import apply_medical_action


class MedicalImageEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.action_space = spaces.Discrete(9)

        # Compact state:
        # [step_ratio, p_initial, p_current, last_action_norm, confidence_gain]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.raw_transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

        self.dataset = ImageFolder(
            "../data_resplit/train",
            transform=self.raw_transform
        )

        self.model = get_model().to(self.device)

        self.model.load_state_dict(
            torch.load(
                "../models/best_baseline.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        self.current_image = None
        self.current_label = None
        self.initial_tensor = None
        self.current_tensor = None

        self.p_initial = 0.0
        self.p_current = 0.0
        self.last_action = 0

        self.step_count = 0
        self.max_actions = 4
        self.action_penalty = 0.01

    def _confidence(self, tensor, label):

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0, label].item()

        return confidence

    def _get_state(self):

        confidence_gain = self.p_current - self.p_initial

        state = np.array(
            [
                self.step_count / self.max_actions,
                self.p_initial,
                self.p_current,
                self.last_action / 8.0,
                confidence_gain
            ],
            dtype=np.float32
        )

        return state

    def reset(self, seed=None, options=None):

        random_index = np.random.randint(
            0,
            len(self.dataset)
        )

        image, label = self.dataset[random_index]

        self.current_image = image
        self.current_label = int(label)

        self.initial_tensor = self.base_transform(image)
        self.current_tensor = self.initial_tensor.clone()

        self.p_initial = self._confidence(
            self.initial_tensor,
            self.current_label
        )

        self.p_current = self.p_initial
        self.last_action = 0
        self.step_count = 0

        return self._get_state(), {}

    def step(self, action):

        action = int(action)
        self.last_action = action

        terminated = False

        if action == 0:
            terminated = True

        else:
            self.current_image = apply_medical_action(
                self.current_image,
                action
            )

            self.current_tensor = self.base_transform(
                self.current_image
            )

            self.p_current = self._confidence(
                self.current_tensor,
                self.current_label
            )

            self.step_count += 1

            if self.step_count >= self.max_actions:
                terminated = True

        truncated = False

        reward = 0.0

        if terminated:

            reward, p_initial, p_final = final_confidence_reward(
                model=self.model,
                initial_tensor=self.initial_tensor,
                final_tensor=self.current_tensor,
                label=self.current_label,
                device=self.device,
                num_actions=self.step_count,
                action_penalty=self.action_penalty
            )

            self.p_current = p_final

        info = {
            "action": action,
            "step_count": int(self.step_count),
            "reward": float(reward),
            "p_initial": float(self.p_initial),
            "p_final": float(self.p_current),
            "label": int(self.current_label)
        }

        return self._get_state(), float(reward), terminated, truncated, info