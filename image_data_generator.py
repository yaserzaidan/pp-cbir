from pathlib import Path

import numpy as np

from image_encryption import ImageEncryption
import tensorflow as tf

class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, subset, batch_size=32, image_shape:tuple=(224, 224), normalization:dict=None, m=4, workers=2, shuffle=True):
        super().__init__(use_multiprocessing=True, workers=workers)
        self.data_dir = Path(data_dir) / subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.m = m
        self.encryption = ImageEncryption(m)
        self.normalization = normalization
        self.image_shape = image_shape

        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.filenames = []
        self.labels = []
        for cls_name in self.classes:
            cls_path = self.data_dir / cls_name
            for img_path in cls_path.glob('*'):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}:
                    self.filenames.append(img_path)
                    self.labels.append(self.class_indices[cls_name])

        self.indices = np.arange(len(self.filenames))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.filenames))
        batch_indices = self.indices[start_idx:end_idx]

        batch_x = np.zeros((len(batch_indices), self.image_shape[0], self.image_shape[1], 3))
        batch_y = np.zeros(len(batch_indices))

        for i, idx in enumerate(batch_indices):
            img = tf.keras.preprocessing.image.load_img(self.filenames[idx], target_size=self.image_shape)
            img = tf.keras.preprocessing.image.img_to_array(img)

            # Apply encryption
            encrypted_img, _, _ = self.encryption.image_encryption(img)

            # Normalize
            encrypted_img = encrypted_img.astype(np.float32) / 255.0
            if self.normalization:
                encrypted_img = (encrypted_img - self.normalization['mean'] ) / self.normalization['std']

            batch_x[i] = encrypted_img
            batch_y[i] = self.labels[idx]

        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
