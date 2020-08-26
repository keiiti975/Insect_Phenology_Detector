import cv2
import torch
import torch.utils.data as data
import imgaug.augmenters as iaa

class insects_dataset(data.Dataset):
    
    def __init__(self, images, labels=None, training=False, method_aug=None):
        """
            init function
            Args:
                - images: np.array, insect images
                - labels: np.array, insect labels
                - training: bool
                - method_aug: [str, ...], sequence of method name
                    possible choices = [
                        HorizontalFlip, VerticalFlip, Rotate]
        """
        self.images = images
        self.labels = labels
        self.training = training
        self.method_aug = method_aug
        
        if training is True:
            if method_aug is not None:
                print("augment == method_aug")
                print("---")
                self.aug_seq = self.create_aug_seq()
                print("---")
            else:
                print("augment == None")
                self.aug_seq = None
        else:
            self.aug_seq = None
        
    def __getitem__(self, index):
        # normalize
        image = self.images[index].astype("float32")
        image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # adopt augmentation
        if self.aug_seq is not None:
            image_aug = self.aug_seq(image=image)
        else:
            image_aug = image
        
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
            elif augmentation == "Rotate":
                print("Rotate")
                aug_list.append(iaa.Rotate((-90, 90)))
            elif augmentation == "Contrast":
                print("Contrast")
                aug_list.append(iaa.LinearContrast((0.5, 1.5)))
            elif augmentation == "Sharpen":
                print("Sharpen")
                aug_list.append(iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)))
            elif augmentation == "Invert":
                print("Invert")
                aug_list.append(iaa.Invert(0.5))
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
            else:
                print("not implemented!: insects_dataset.create_aug_seq")
        
        aug_seq = iaa.SomeOf((0, 2), aug_list, random_order=True)
        return aug_seq