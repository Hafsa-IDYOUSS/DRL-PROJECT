import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torchvision.transforms as transforms


def get_baseline_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def apply_medical_action(image, action):
    action = int(action)

    # 0 = STOP / no operation
    if action == 0:
        return image

    # 1 = Contrast +
    if action == 1:
        return ImageEnhance.Contrast(image).enhance(1.25)

    # 2 = Contrast -
    if action == 2:
        return ImageEnhance.Contrast(image).enhance(0.85)

    # 3 = Brightness +
    if action == 3:
        return ImageEnhance.Brightness(image).enhance(1.15)

    # 4 = Brightness -
    if action == 4:
        return ImageEnhance.Brightness(image).enhance(0.90)

    # 5 = CLAHE
    if action == 5:
        img_np = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        img_np = clahe.apply(img_np)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    # 6 = Denoise
    if action == 6:
        img_np = np.array(image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np,
            None,
            h=5,
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return Image.fromarray(denoised)

    # 7 = Sharpen
    if action == 7:
        return image.filter(ImageFilter.SHARPEN)

    # 8 = Gamma Correction
    if action == 8:
        gamma = 0.85
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_np = np.power(img_np, gamma)
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    return image


class MedicalActionTransform:
    def __init__(self, action):
        self.action = int(action)

    def __call__(self, image):
        return apply_medical_action(image, self.action)


class MedicalSequenceTransform:
    def __init__(self, sequence):
        self.sequence = [int(a) for a in sequence]

    def __call__(self, image):
        for action in self.sequence:
            if action == 0:
                break
            image = apply_medical_action(image, action)
        return image


def get_transform_by_action(action):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        MedicalActionTransform(action),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_transform_by_sequence(sequence):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        MedicalSequenceTransform(sequence),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_target_domain_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        MedicalSequenceTransform([1, 5, 8]),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])