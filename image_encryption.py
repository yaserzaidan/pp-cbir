
from pathlib import Path
import tensorflow as tf
import random
import cv2
import numpy as np


class ImageEncryption:
    def __init__(self, m):
        self.m = m

    def image_encryption(self, image):
        # Convert to numpy if tensor
        if isinstance(image, tf.Tensor):
            image = image.numpy()

        img = cv2.resize(image, (224, 224))
        height, width = img.shape[:2]
        w_subblock = width // self.m
        h_subblock = height // self.m

        encrypted_image = np.zeros_like(img)
        img_sub_key = []
        sub_keys = []

        for i in range(self.m):
            for j in range(self.m):
                x1, x2 = i * w_subblock, (i + 1) * w_subblock
                y1, y2 = j * h_subblock, (j + 1) * h_subblock
                subblock = img[y1:y2, x1:x2]

                encrypted_subblock, subK = self.intrablock_texture_encryption(subblock)
                encrypted_subblock = self.color_encryption(encrypted_subblock, i * self.m + j)

                encrypted_image[y1:y2, x1:x2] = encrypted_subblock
                sub_keys.append(subK)
                img_sub_key.append((i, j))

        random.shuffle(img_sub_key)
        return encrypted_image.astype(np.uint8), img_sub_key, sub_keys

    def intrablock_texture_encryption(self, subblock):
        lr, lg, lb = cv2.split(subblock)

        keys = {}
        for channel, color in zip([lr, lg, lb], ['r', 'g', 'b']):
            key = np.arange(channel.size)
            np.random.shuffle(key)
            channel.flat[:] = channel.flat[key]
            keys[color] = key

        encrypted_subblock = cv2.merge((lr, lg, lb))
        return encrypted_subblock, keys

    def color_encryption(self, subblock, i):
        er, eg, eb = cv2.split(subblock)

        eb = ((eb + ((i / self.m) * 256 / self.m)) % 256).astype(np.uint8)
        er = ((er + ((i % self.m) * 256 / self.m)) % 256).astype(np.uint8)
        eg = ((eg + ((i / self.m + i % self.m) * 128 / self.m)) % 256).astype(np.uint8)

        encrypted_subblock = cv2.merge((er, eg, eb))
        return encrypted_subblock
    
    def image_decryption(self, encrypted_image, img_sub_key, sub_keys):
        height, width = encrypted_image.shape[:2]
        w_subblock = width // self.m
        h_subblock = height // self.m

        decrypted_image = np.zeros_like(encrypted_image)
        reverse_key = {index: (i, j) for index, (i, j) in enumerate(img_sub_key)}

        for index, subK in enumerate(sub_keys):
            i, j = reverse_key[index]
            x1, x2 = i * w_subblock, (i + 1) * w_subblock
            y1, y2 = j * h_subblock, (j + 1) * h_subblock
            subblock = encrypted_image[y1:y2, x1:x2]

            subblock = self.color_decryption(subblock, i * self.m + j)
            subblock = self.intrablock_texture_decryption(subblock, subK)

            decrypted_image[y1:y2, x1:x2] = subblock

        return decrypted_image.astype(np.uint8)

    def intrablock_texture_decryption(self, subblock, keys):
        lr, lg, lb = cv2.split(subblock)

        for channel, color in zip([lr, lg, lb], ['r', 'g', 'b']):
            key = keys[color]
            reverse_key = np.argsort(key)
            channel.flat[:] = channel.flat[reverse_key]

        decrypted_subblock = cv2.merge((lr, lg, lb))
        return decrypted_subblock

    def color_decryption(self, subblock, i):
        er, eg, eb = cv2.split(subblock)

        eb = ((eb - ((i / self.m) * 256 / self.m)) % 256).astype(np.uint8)
        er = ((er - ((i % self.m) * 256 / self.m)) % 256).astype(np.uint8)
        eg = ((eg - ((i / self.m + i % self.m) * 128 / self.m)) % 256).astype(np.uint8)

        decrypted_subblock = cv2.merge((er, eg, eb))
        return decrypted_subblock
