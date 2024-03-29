import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import torch
import torch.utils.data as data
import imgaug.augmenters as iaa
# evaluation
from evaluation.classification.statistics import get_size_list_from_xte


class insects_dataset(data.Dataset):
    
    def __init__(self, images, labels=None, training=False, method_aug=None, size_normalization=None):
        """
            init function
            Args:
                - images: np.array, insect images
                - labels: np.array, insect labels
                - training: bool
                - method_aug: [str, ...], sequence of method name
                    possible choices = [
                        HorizontalFlip, VerticalFlip, Rotate]
                - size_normalization: str, choice [None, "mu", "sigma", "mu_sigma", "uniform"]
        """
        self.images = images
        self.labels = labels
        self.training = training
        self.method_aug = method_aug
        self.size_normalization = size_normalization
        
        if training is True:
            if method_aug is not None:
                print("augment == method_aug")
                print("---")
                self.aug_seq = self.create_aug_seq()
                print("---")
            else:
                print("augment == None")
                self.aug_seq = None
            
            if size_normalization in ["mu", "sigma", "mu_sigma", "uniform"]:
                print("size_normalization == {}".format(size_normalization))
                if size_normalization == "uniform":
                    self.resize_px = 50
                    print("resize_px == {}".format(self.resize_px))
                insect_size_list, insect_size_dic = self.get_insect_size(images, labels)
                mu, sigma = self.calc_mu_sigma(insect_size_dic)
                self.insect_size_list = np.log2(insect_size_list)
                self.insect_size_dic = insect_size_dic
                self.mu = mu
                self.sigma = sigma
            else:
                print("size_normalization == None")
        else:
            self.aug_seq = None
        
    def __getitem__(self, index):
        image = self.images[index].astype("uint8")
        
        # adopt size normalization
        if self.training and self.size_normalization in ["mu", "sigma", "mu_sigma", "uniform"]:
            image = self.adopt_size_normalization(image, self.labels[index], self.insect_size_list[index])
        
        # adopt augmentation
        if self.aug_seq is not None:
            image_aug = self.aug_seq(image=image)
        else:
            image_aug = image
            
        # normalize
        image_aug = image_aug.astype("float32")
        image_aug = cv2.normalize(image_aug, image_aug, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # create pytorch image
        image_aug = image_aug.transpose(2,0,1).astype("float32")
        image_aug = torch.from_numpy(image_aug).clone()
        
        if self.training is True:
            label = self.labels[index]
            return image_aug, label
        else:
            return image_aug
    
    def __len__(self):
        return self.images.shape[0]
    
    def get_insect_size(self, X, Y):
        """
            get list, dictionary of label to size
            Args:
                - X: np.array, shape==[insect_num, height, width, channels]
                - Y: np.array, shape==[insect_num]
        """
        idx = np.unique(Y)
        X_size = np.array(get_size_list_from_xte(X))
        insect_size_dic = {}
        for i in range(len(idx)):
            insect_filter = Y == i
            filtered_X_size = X_size[insect_filter]
            filtered_X_size = np.sort(filtered_X_size)
            insect_size_dic.update({i: filtered_X_size})
        return X_size, insect_size_dic
    
    def calc_mu_sigma(self, insect_size_dic):
        """
            calculate mu, sigma for each insect size distribution
            Args:
                - insect_size_dic: dict, {label: size_array}
        """
        gmm = GMM(n_components=1, covariance_type="spherical")
        mu = []
        sigma = []
        for key, value in insect_size_dic.items():
            x = np.log2(insect_size_dic[key])
            gmm.fit(x.reshape(-1, 1))
            mu.append(gmm.means_.reshape([-1])[0])
            sigma.append(np.sqrt(gmm.covariances_)[0])
        return np.array(mu), np.array(sigma)
    
    def adopt_size_normalization(self, image, label, size):
        """
            adopt image to size normalization
            Args:
                - image: PIL.Image
                - label: int
                - size: int
                
            - mu:
                convert size distribution => mu = mu_average, sigma = keep
                Formula:
                    2 ** (mu_average - mu_each)
            - sigma:
                convert size distribution => mu = keep, sigma = 1
                Formula:
                    2 ** ((1 - sigma_each) / sigma_each * (x - mu_each))
            - mu_sigma:
                convert size distribution => mu = mu_average, sigma = 1
                Formula:
                    2 ** ((1 - sigma_each) / sigma_each) * 
                    2 ** ((mu_average * sigma_each - mu_each) / sigma_each)
            - uniform:
                random resize and padding
        """
        mu_average = self.mu.mean()
        size_norm_augs = []
        if self.size_normalization == "mu":
            for mu_each, sigma_each in zip(self.mu, self.sigma):
                correction_term = mu_average - mu_each
                size_norm_augs.append(
                    iaa.Affine(scale=(np.sqrt(2 ** correction_term), np.sqrt(2 ** correction_term)))
                )
        elif self.size_normalization == "sigma":
            for mu_each, sigma_each in zip(self.mu, self.sigma):
                correction_term = (1 - sigma_each) / sigma_each * (size - mu_each)
                size_norm_augs.append(
                    iaa.Affine(scale=(np.sqrt(2 ** correction_term), np.sqrt(2 ** correction_term)))
                )
        elif self.size_normalization == "mu_sigma":
            for mu_each, sigma_each in zip(self.mu, self.sigma):
                correction_term = ((1 - sigma_each) * size - mu_each + mu_average * sigma_each) / sigma_each
                size_norm_augs.append(
                    iaa.Affine(scale=(np.sqrt(2 ** correction_term), np.sqrt(2 ** correction_term)))
                )
        elif self.size_normalization == "uniform":
            for mu_each, sigma_each in zip(self.mu, self.sigma):
                size_norm_augs.append(
                    iaa.CropAndPad(
                        px=(-1 * self.resize_px, 0),
                        sample_independently=False
                    )
                )
        else:
            pass
        
        normed_image = size_norm_augs[label](image=image)
        return normed_image
    
    def create_aug_seq(self):
        aug_list = []
        # create augmentation
        for augmentation in self.method_aug:
            if augmentation == "HorizontalFlip":
                print("HorizontalFlip")
                aug_list.append(iaa.Fliplr(0.5))
            elif augmentation == "VerticalFlip":
                print("VerticalFlip")
                aug_list.append(iaa.Flipud(0.5))
            elif augmentation == "CropandResize":
                print("CropandResize")
                aug_list.append(iaa.KeepSizeByResize(
                                    iaa.OneOf([
                                        iaa.Crop((int(200/2), int(200/2)), keep_size=False),
                                        iaa.Crop((int(200/3 * 2), int(200/3 * 2)), keep_size=False),
                                        iaa.Crop((int(200/4 * 3), int(200/4 * 3)), keep_size=False)
                                    ]),
                                    interpolation=cv2.INTER_NEAREST
                                ))
            elif augmentation == "CLAHE":
                print("CLAHE")
                aug_list.append(iaa.CLAHE())
            elif augmentation == "Sharpen":
                print("Sharpen")
                aug_list.append(iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.0, 1.0)))
            elif augmentation == "Emboss":
                print("Emboss")
                aug_list.append(iaa.Emboss(alpha=(0.0, 1.0), strength=(0.0, 1.0)))
            elif augmentation == "Shear":
                print("Shear")
                aug_list.append(iaa.OneOf([
                                    iaa.ShearX((-20, 20)),
                                    iaa.ShearY((-20, 20))
                                ]))
            elif augmentation == "Translate":
                print("Translate")
                aug_list.append(iaa.OneOf([
                                    iaa.TranslateX(px=(-20, 20)),
                                    iaa.TranslateY(px=(-20, 20))
                                ]))
            elif augmentation == "Rotate":
                print("Rotate")
                aug_list.append(iaa.Rotate((-90, 90)))
            elif augmentation == "AutoContrast":
                print("AutoContrast")
                aug_list.append(iaa.pillike.Autocontrast())
            elif augmentation == "Invert":
                print("Invert")
                aug_list.append(iaa.Invert(0.5))
            elif augmentation == "Equalize":
                print("Equalize")
                aug_list.append(iaa.pillike.Equalize())
            elif augmentation == "Solarize":
                print("Solarize")
                aug_list.append(iaa.Solarize(0.5, threshold=(32, 128)))
            elif augmentation == "Posterize":
                print("Posterize")
                aug_list.append(iaa.color.Posterize())
            elif augmentation == "Contrast":
                print("Contrast")
                aug_list.append(iaa.pillike.EnhanceContrast())
            elif augmentation == "Color":
                print("Color")
                aug_list.append(iaa.pillike.EnhanceColor())
            elif augmentation == "Brightness":
                print("Brightness")
                aug_list.append(iaa.pillike.EnhanceBrightness())
            elif augmentation == "Sharpness":
                print("Sharpness")
                aug_list.append(iaa.pillike.EnhanceSharpness())
            elif augmentation == "Cutout":
                print("Cutout")
                aug_list.append(iaa.Cutout(nb_iterations=1))
            elif augmentation == "All":
                # adjust to AutoAugment
                print("All")
                aug_list = [
                    iaa.OneOf([
                        iaa.ShearX((-20, 20)),
                        iaa.ShearY((-20, 20))
                    ]),
                    iaa.OneOf([
                        iaa.TranslateX(px=(-20, 20)),
                        iaa.TranslateY(px=(-20, 20))
                    ]),
                    iaa.Rotate((-90, 90)),
                    iaa.pillike.Autocontrast(),
                    iaa.Invert(0.5),
                    iaa.pillike.Equalize(),
                    iaa.Solarize(0.5, threshold=(32, 128)),
                    iaa.color.Posterize(),
                    iaa.pillike.EnhanceContrast(),
                    iaa.pillike.EnhanceColor(),
                    iaa.pillike.EnhanceBrightness(),
                    iaa.pillike.EnhanceSharpness(),
                    iaa.Cutout(nb_iterations=1),
                ]
            elif augmentation == "All_plus_alpha":
                # use All Augment
                print("All_plus_alpha")
                aug_list = [
                    iaa.OneOf([
                        iaa.ShearX((-20, 20)),
                        iaa.ShearY((-20, 20))
                    ]),
                    iaa.OneOf([
                        iaa.TranslateX(px=(-20, 20)),
                        iaa.TranslateY(px=(-20, 20))
                    ]),
                    iaa.Rotate((-90, 90)),
                    iaa.pillike.Autocontrast(),
                    iaa.Invert(0.5),
                    iaa.pillike.Equalize(),
                    iaa.Solarize(0.5, threshold=(32, 128)),
                    iaa.color.Posterize(),
                    iaa.pillike.EnhanceContrast(),
                    iaa.pillike.EnhanceColor(),
                    iaa.pillike.EnhanceBrightness(),
                    iaa.pillike.EnhanceSharpness(),
                    iaa.Cutout(nb_iterations=1),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CLAHE(),
                    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.0, 1.0)),
                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.0, 1.0)),
                ]
            elif augmentation == "Coordinate":
                # use coordinate transform only
                print("Coordinate")
                aug_list = [
                    iaa.OneOf([
                        iaa.ShearX((-20, 20)),
                        iaa.ShearY((-20, 20))
                    ]),
                    iaa.OneOf([
                        iaa.TranslateX(px=(-20, 20)),
                        iaa.TranslateY(px=(-20, 20))
                    ]),
                    iaa.Rotate((-90, 90)),
                    iaa.CropAndPad(
                        px=(-1 * 60, 60),
                        sample_independently=False
                    ),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                ]
            elif augmentation == "RandomResize":
                print("RandomResize")
                # imgaug function
                """
                aug_list.append(
                    iaa.CropAndPad(
                        px=(-1 * 50, 0),
                        sample_independently=False
                    )
                )
                """
                # self coded function
                aug_list.append(
                    iaa.Lambda(
                        func_images=aug_scale
                    )
                )
            elif augmentation == "Jigsaw":
                print("Jigsaw")
                aug_list.append(iaa.Jigsaw(nb_rows=4, nb_cols=4))
            else:
                print("not implemented!: insects_dataset.create_aug_seq")
        
        aug_seq = iaa.SomeOf((5, 6), aug_list, random_order=True)
        return aug_seq
    
    
class maha_insects_dataset(data.Dataset):
    
    def __init__(self, images, labels):
        """
            init function
            Args:
                - images: np.array, insect images
                - labels: np.array, insect labels
        """
        self.images = images
        self.labels = labels
        self.sample()
        
    def __getitem__(self, index):
        image = self.sampled_images[index].astype("uint8")
            
        # normalize
        image = image.astype("float32")
        image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # create pytorch image
        image = image.transpose(2,0,1).astype("float32")
        image = torch.from_numpy(image).clone()
        
        return image
    
    def __len__(self):
        return self.sampled_images.shape[0]
    
    def sample(self, sample_num=30):
        sampled_images = []
        sampled_labels = []
        idx, counts = np.unique(self.labels, return_counts=True)
        for i in idx:
            sampled_id = np.random.choice(np.where(self.labels == i)[0], size=sample_num)
            sampled_images.extend(self.images[sampled_id])
            sampled_labels.extend(self.labels[sampled_id])
        self.sampled_images = np.array(sampled_images)
        self.sampled_labels = np.array(sampled_labels)
        
              
def aug_scale(images, random_state, parents, hooks):
    image_array = []
    for img in images:
        ratio = np.random.rand() + 1.0  # 1.0 <= ratio < 2.0
        img = img.astype("float32")
        
        h, w = img.shape[:2]
        src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
        dest = src * ratio
        h_diff = (ratio - 1) * (h / 2)
        w_diff = (ratio - 1) * (w / 2)
        dest[:, 0] -= w_diff
        dest[:, 1] -= h_diff
        affine = cv2.getAffineTransform(src, dest)
        
        img = cv2.warpAffine(img, affine, (200, 200), cv2.INTER_LANCZOS4)
        image_array.append(img)
    images = np.array(image_array).astype("uint8")
    return images