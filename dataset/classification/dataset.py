import cv2
import torch
import torch.utils.data as data
import imgaug.augmenters as iaa

class insects_dataset(data.Dataset):
    
    def __init__(self, images, labels, training=False, method_aug=None):
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
        
        if training is True and method_aug is not None:
            print("augment == method_aug")
            print("---")
            self.aug_seq = self.create_aug_seq()
            print("---")
        else:
            print("augment == None")
            self.aug_seq = None
        
    def __getitem__(self, index):
        # adopt augmentation
        if self.aug_seq is not None:
            image_aug = self.aug_seq(image=self.images[index])
        else:
            image_aug = self.images[index]
        
        # normalize
        image_aug = image_aug.astype("float32")
        image_aug = cv2.normalize(image_aug, image_aug, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # create pytorch image
        image_aug = image_aug.transpose(2,0,1).astype("float32")
        image_aug = torch.from_numpy(image_aug).clone()
        
        label = self.labels[index]
        return image_aug, label
    
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
            else:
                print("not implemented!: insects_dataset.create_aug_seq")
        
        aug_seq = iaa.Sequential(aug_list)
        return aug_seq