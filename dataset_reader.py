import random
from pathlib import Path
import numpy as np

class DatasetReader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_indices = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.filenames = {}

        for class_name in self.classes:
            self.filenames[class_name] = []

        for cls_name in self.classes:
            cls_path = self.data_dir / cls_name
            for img_path in cls_path.glob('*'):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif'}:
                    self.filenames[cls_name].append((img_path,self.class_indices[cls_name]))

        self.indices = np.arange(len(self.filenames))

    def get_random_images_from_class(self, class_index: int=None, images_number:int=5):
        class_name = None
        for name, idx in self.class_indices.items():
            if idx == class_index:
                class_name = name
        if not class_name:
            raise Exception(f"class {class_index} doesn't exist in dataset")
        images_dicts = random.sample(self.filenames[class_name], len(self.filenames[class_name]))
        image_paths_list = []
        for image_path, _ in images_dicts[:images_number]:
            image_paths_list.append(image_path)

        return image_paths_list
