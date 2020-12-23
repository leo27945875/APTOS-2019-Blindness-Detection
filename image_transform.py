import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image,  ImageChops


class Trim(object):
    def __call__(self, image: Image) -> Image:
        bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        else:
            return image


class CropImageFromGray:
    def __call__(self, img: np.array, tol: int=7) -> np.array:
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]

        elif img.ndim == 3:
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = grayImg > tol
            row = mask.any(1)
            col = mask.any(0)
            checkShape = img[:, :, 0][np.ix_(row, col)].shape[0]
            if (checkShape == 0):
                return img

            else:
                img1 = img[:, :, 0][np.ix_(row, col)]
                img2 = img[:, :, 1][np.ix_(row, col)]
                img3 = img[:, :, 2][np.ix_(row, col)]
                img  = np.stack([img1, img2, img3], axis=-1)
            
            return img


class ClearImage:
    def __call__(self, img: np.array, sigmaX: int =10.):
        return cv2.addWeighted (img, 4, cv2.GaussianBlur(img, (0, 0) ,sigmaX) ,-4 ,128)