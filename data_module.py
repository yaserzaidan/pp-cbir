from image_data_generator import ImageDataGenerator
from collections import Counter



class DataModule:
    def __init__(self, data_dir, batch_size=32, m=2, image_shape:tuple=(224,224), normalization:dict=None):
        self.data_dir = data_dir
        self.m = m
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.normalization = normalization

        self.train_generator = ImageDataGenerator(
            data_dir=self.data_dir,
            subset='train',
            batch_size=self.batch_size,
            normalization=self.normalization,
            image_shape=self.image_shape,
            m=self.m,
            shuffle=True
        )

        self.test_generator = ImageDataGenerator(
            data_dir=self.data_dir,
            subset='test',
            batch_size=self.batch_size,
            normalization=self.normalization,
            image_shape=self.image_shape,
            m=self.m,
            shuffle=False
        )

        self.valid_generator = ImageDataGenerator(
            data_dir=self.data_dir,
            subset='valid',
            batch_size=self.batch_size,
            normalization=self.normalization,
            image_shape=self.image_shape,
            m=self.m,
            shuffle=False
        )

    def calculate_class_weights(self):
        # Calculate class weights
        class_counts = Counter(self.train_generator.labels)
        total_samples = len(self.train_generator.labels)
        return {
            i: total_samples / count
            for i, count in class_counts.items()
        }