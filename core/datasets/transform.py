import random

import numpy as np
import numbers
import collections
from PIL import Image

import torchvision
from torchvision.transforms import functional as F
import cv2
from skimage.transform import warp, AffineTransform

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), F.to_tensor(label).squeeze()

class ToNumpy(object):
    def __call__(self, image, label):
        return image.numpy().transpose(1, 2, 0), label.numpy()

class ChangeToPIL(object):
    def __init__(self):
        self.pil_trans = ToPILImage()

    def __call__(self, image, label):
        return self.pil_trans(image), self.pil_trans(label)


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, label):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, resize_label=True):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        image = F.resize(image, self.size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray): 
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label,(self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, self.size, Image.NEAREST)
        return image, label

class RandomScale(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, scale, size=None, resize_label=True):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        self.scale = scale
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        w, h = image.size
        if self.size:
            h, w = self.size
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        size = (int(h*temp_scale), int(w*temp_scale))
        image = F.resize(image, size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label,(self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, size, Image.NEAREST)
        return image, label

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, label_fill=255, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        if isinstance(size, numbers.Number):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(size, tuple):
            if padding is not None and len(padding)==2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            else:
                self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.label_fill = label_fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((self.padding[1], self.padding[3]), (self.padding[0], self.padding[2]), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, self.padding, self.label_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((0, 0), (self.size[1]-img.size[0], self.size[1]-img.size[0]), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, (self.size[1] - lab.size[0], 0), self.label_fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((self.size[0]-img.size[1], self.size[0]-img.size[1]), (0, 0), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, (0, self.size[0] - lab.size[1]), self.label_fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if isinstance(lab, np.ndarray):
            # assert the shape of label is in the order of (h, w, c)
            lab = lab[i:i+h, j:j+w, :]
        else:
            lab = F.crop(lab, i, j, h, w)
        return img, lab

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            if isinstance(label, np.ndarray): 
                # assert the shape of label is in the order of (h, w, c)
                label = label[:,::-1,:]
            else:
                label = F.hflip(label)
        return image, label


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, label):
        image = self.color_jitter(image)
        return image, label

class ResizeAndPad(object):
    """
    Resize the largest side of the image to the given size and pad the rest with ignore_index.
    """
    def __init__(self, size, ignore_label=255):
        self.size = size  # size: (h, w)
        self.crop_h, self.crop_w = size
        self.ignore_label=ignore_label

    def __call__(self, image, label):
        image = np.array(image)
        label = np.array(label)
        assert image.shape[:2] == label.shape
        img_h, img_w = label.shape

        # Resize
        if img_h > img_w and img_h > self.crop_h:
            new_w = int((self.crop_h / img_h) * img_w)
            img_h, img_w = self.crop_h, new_w
            image = cv2.resize(image, (new_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        elif img_w >= img_h and img_w > self.crop_w:
            new_h = int((self.crop_w / img_w) * img_h)
            img_h, img_w = new_h, self.crop_w
            image = cv2.resize(image, (self.crop_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.crop_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(self.ignore_label,))
        return image, label


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range=None,
                 background_val=1,
                 ignore_label=False
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        self.cval = background_val
        self.ignore_label = ignore_label

    def calc_new_shape(self, af_trans, im_shape):
        tl = np.array([0, 0])
        tr = np.array([0, im_shape[1]])
        bl = np.array([im_shape[0], 0])
        br = np.array([im_shape[0], im_shape[1]])

        new_coords = af_trans(np.array([tl[::-1], tr[::-1], bl[::-1], br[::-1]]))  # af trans takes cooridnates (x, y) rather than (y, x)
        new_tl, new_tr, new_bl, new_br = new_coords[0], new_coords[1], new_coords[2], new_coords[3]
        new_h = np.ceil(np.max([new_tl[1], new_tr[1], new_bl[1], new_br[1]]) - np.min([new_tl[1], new_tr[1], new_bl[1], new_br[1]]))
        new_w = np.ceil(np.max([new_tl[0], new_tr[0], new_bl[0], new_br[0]]) - np.min([new_tl[0], new_tr[0], new_bl[0], new_br[0]]))

        return new_h, new_w

    def calc_center_translation(self, af_mat, trans_shape):
        padded_center = np.array([trans_shape[1] // 2, trans_shape[0] // 2])
        new_center = af_mat(padded_center)[0]

        trans_w = padded_center[0] - new_center[0]
        trans_h = padded_center[1] - new_center[1]

        return trans_w, trans_h

    def __call__(self, img, label, scale_range=None):
        if img is None:
            return None, None
        img_data = np.array(img)
        if not self.ignore_label and label is not None:
            label = np.array(label)
        h, w, n_chan = img_data.shape

        # Create Affine Transform without translation
        if scale_range is not None:
            scale_min, scale_max = self.scale_range
            if scale_range[0] is not None:
                scale_min = scale_range[0]
            if scale_range[1] is not None:
                scale_max = scale_range[1]
            self.scale_range = (scale_min, scale_max)
        scale_h = np.random.uniform(*self.scale_range)
        scale_w = np.random.uniform(*self.scale_range)
        mirror = bool(np.random.randint(0, 2))
        if mirror:
            scale_w *= -1
        scale = (scale_w, scale_h)

        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)

        af = AffineTransform(scale=scale, rotation=rotation, shear=shear)

        # Calculate new shape and Pad Image (If needed)
        new_shape = self.calc_new_shape(af, img_data.shape)
        addition_h = new_shape[0] - h
        addition_w = new_shape[1] - w
        pad_h = max(int(addition_h // 2), 0)
        pad_w = max(int(addition_w // 2), 0)
        img_data = np.pad(img_data, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)])
        if not self.ignore_label and label is not None:
            label = np.pad(label, [(pad_h, pad_h), (pad_w, pad_w)])

        # Calculate translation and creating new affine transformation
        translation = self.calc_center_translation(af, img_data.shape)
        af = AffineTransform(scale=scale, rotation=rotation, shear=shear, translation=translation)

        # Warp Image
        img_data1 = warp(img_data, af.inverse)
        if not self.ignore_label and label is not None:
            label = warp(label / 255, af.inverse, order=0, cval=self.cval) * 255

        # Crop Image (if needed)
        crop_h = max(-int(addition_h // 2), 0)
        crop_w = max(-int(addition_w // 2), 0)
        img_data1 = img_data1[crop_h: img_data1.shape[0]-crop_h, crop_w: img_data1.shape[1]-crop_w, :]
        img1 = np.uint8(img_data1 * 255)
        # img1 = Image.fromarray(np.uint8(img_data1 * 255))
        if not self.ignore_label and label is not None:
            label = label[crop_h: label.shape[0]-crop_h, crop_w: label.shape[1]-crop_w]
            # label = np.uint8(label)
            # label = Image.fromarray(np.uint8(label * 255))

        if self.ignore_label:
            return Image.fromarray(img1), Image.fromarray(img1)
        elif label is not None:
            return Image.fromarray(img1), Image.fromarray(label)
        else:
            return Image.fromarray(img1), None
