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
        image = self.images[index].astype("uint8")
        
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
                print("All")
                aug_list.append(iaa.SomeOf((1, 2), [
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
                                    iaa.CLAHE(),
                                    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.0, 1.0)),
                                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.0, 1.0)),
                                    iaa.Fliplr(0.5),
                                    iaa.Flipud(0.5)
                                ], random_order=True))
            else:
                print("not implemented!: insects_dataset.create_aug_seq")
        
        aug_seq = iaa.SomeOf((0, 1), aug_list, random_order=True)
        return aug_seq