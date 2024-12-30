from cbir_model import CBIRModel
from image_encryption import ImageEncryption
from dataset_reader import DatasetReader

from pathlib import Path
import numpy as np
import tensorflow as tf

class CBIRSystem:
    def __init__(self, cbir_model: CBIRModel, database_dir: Path, data_module=None):
        self.database_dir = database_dir
        self.is_cbir = True


        if isinstance(cbir_model, CBIRModel):
            self.cbir_model = cbir_model.model
            self.data_module = cbir_model.data_module
        else:
            self.cbir_model = cbir_model
            self.data_module = data_module
            self.is_cbir = False

        self.encryptor = ImageEncryption(self.data_module.m)

        if self.database_dir.exists():
            self.dataset_reader = DatasetReader(self.database_dir)
        else:
            raise Exception(f"Database directory path provided doesn't exist : {self.database_dir}")

    def query_similar_images(self, image, k):
        if len(image.shape) != 4:
            image = np.expand_dims(image, axis=0)

        if not self.is_cbir:
            pred = self.cbir_model.predict_image(image)
        else:
            pred = self.cbir_model.predict(image, verbose=0)

        predicted_image_class = np.argmax(pred)

        similar_images_paths = self.dataset_reader.get_random_images_from_class(predicted_image_class, k)
        similar_images = []
        for image_path in similar_images_paths:
            query_img_and_keys = self.read_and_encrypt_image(image_path)
            similar_images.append(query_img_and_keys)

        return similar_images, predicted_image_class

    def read_and_encrypt_image(self, image_path: Path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Apply encryption
        encrypted_img, img_sub_key, sub_keys = self.encryptor.image_encryption(img)

        # Normalize
        encrypted_img = encrypted_img.astype(np.float32) / 255.0
        encrypted_img = (encrypted_img - self.data_module.normalization['mean'] ) / self.data_module.normalization['std']

        return encrypted_img, img_sub_key, sub_keys

    def decrypt_image(self, image_and_keys:tuple):
        image, img_sub_key, sub_keys = image_and_keys
        image = image * 255.0
        image = image * self.data_module.normalization['std']
        image = image + self.data_module.normalization['mean']

        return self.encryptor.image_decryption(image, img_sub_key, sub_keys)

    def info(self):
        return f"""
        CBIRSystem :
            core model : {self.cbir_model.name}
            image shape : {self.data_module.image_shape}
            trained with batch size : {self.data_module.batch_size}
            number of sub-blocks : {self.data_module.m}
        """

