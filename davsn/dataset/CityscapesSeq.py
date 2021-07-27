import numpy as np

from advent.utils.serialization import json_load
from davsn.dataset.base_dataset import BaseDataset

class CityscapesSeqDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path='', labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit_sequence' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name_cf = self.files[index]
        label = self.get_labels(label_file)
        label_cf = self.map_labels(label).copy()
        image_cf = self.get_image(img_file)
        image_cf = self.preprocess(image_cf)
        frame_cf = int(name_cf.split('/')[-1].replace('_leftImg8bit.png','')[-6:])
        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - 1).zfill(6) + '_leftImg8bit.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        image_kf = self.get_image(file_kf)
        image_kf = self.preprocess(image_kf.copy())
        return image_cf.copy(), label_cf, image_kf.copy(), np.array(image_cf.shape), name_cf
