from os.path import join as pj
import random
import torchvision
from model.resnet import aa_transforms
from model.resnet import faa_transforms

class AutoAugment(object):
    def __init__(self, policy_dir=None):
        self.policy_dir = policy_dir
        if policy_dir is not None:
            self.policies = read_transform_txt(policy_dir)
        else:
            self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
                ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
                ['Color', 0.4, 3, 'Brightness', 0.6, 7],
                ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
                ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
                ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
                ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
                ['Brightness', 0.9, 6, 'Color', 0.2, 8],
                ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
                ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
                ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
                ['Color', 0.9, 9, 'Equalize', 0.6, 6],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                ['Brightness', 0.1, 3, 'Color', 0.7, 0],
                ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
                ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
            ]

    def __call__(self, img):
        if self.policy_dir is not None:
            img = self.policies[random.randrange(len(self.policies))](img)
        else:
            img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img

operations = {
    'ShearX': lambda img, magnitude: aa_transforms.shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: aa_transforms.shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: aa_transforms.translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: aa_transforms.translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: aa_transforms.rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: aa_transforms.auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: aa_transforms.invert(img, magnitude),
    'Equalize': lambda img, magnitude: aa_transforms.equalize(img, magnitude),
    'Solarize': lambda img, magnitude: aa_transforms.solarize(img, magnitude),
    'Posterize': lambda img, magnitude: aa_transforms.posterize(img, magnitude),
    'Contrast': lambda img, magnitude: aa_transforms.contrast(img, magnitude),
    'Color': lambda img, magnitude: aa_transforms.color(img, magnitude),
    'Brightness': lambda img, magnitude: aa_transforms.brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: aa_transforms.sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: aa_transforms.cutout(img, magnitude),
}

def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img

def read_transform_txt(policy_dir, policy_filename="policy.txt"):
    """
        read transform from txt and create torchvision transform
        Args:
            - policy_dir: str
        Return:
    """
    with open(pj(policy_dir, policy_filename), mode='r') as f:
        lines = f.readlines()
        transform = []
        for line in lines:
            elem_array = line.split(' ')
            policy_array = []
            for i, elem in enumerate(elem_array):
                if i % 3 == 0:
                    if elem == "ShearXY":
                        policy_array.append(faa_transforms.ShearXY(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "TranslateXY":
                        policy_array.append(faa_transforms.TranslateXY(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Rotate":
                        policy_array.append(faa_transforms.Rotate(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "AutoContrast":
                        policy_array.append(faa_transforms.AutoContrast(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Invert":
                        policy_array.append(faa_transforms.Invert(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Equalize":
                        policy_array.append(faa_transforms.Equalize(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Solarize":
                        policy_array.append(faa_transforms.Solarize(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Posterize":
                        policy_array.append(faa_transforms.Posterize(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Contrast":
                        policy_array.append(faa_transforms.Contrast(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Color":
                        policy_array.append(faa_transforms.Color(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Brightness":
                        policy_array.append(faa_transforms.Brightness(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Sharpness":
                        policy_array.append(faa_transforms.Sharpness(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                    if elem == "Cutout":
                        policy_array.append(faa_transforms.Cutout(prob=float(elem_array[i+1]), mag=float(elem_array[i+2])))
                else:
                    continue
            policy_array = torchvision.transforms.Compose(policy_array)
            transform.append(policy_array)
        return transform