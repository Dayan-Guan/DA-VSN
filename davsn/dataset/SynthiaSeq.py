import numpy as np

from advent.utils.serialization import json_load
from davsn.dataset.base_dataset import BaseDataset

class SynthiaSeqDataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        # map to cityscape's ids
        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 5: 3, 7: 4, 15: 5, 9: 6, 6: 7, 1: 8, 10: 9, 11: 10, 8: 11,}

    def get_metadata(self, name):
        img_file = self.root / 'rgb' / name
        label_file = self.root / 'label' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels_synthia_seq(label_file)
        image = image[:-120, :, :]
        label = label[:-120, :]
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        frame = int(name.split('/')[-1].replace('.png',''))
        name1 = name.replace(str(frame).zfill(6) + '.png', str(frame-1).zfill(6) + '.png')
        file1 = self.root / 'rgb' / name1
        image1 = self.get_image(file1)
        image1 = image1[:-120, :, :]
        image1 = self.preprocess(image1.copy())
        return image.copy(), label_copy.copy(), image1.copy(), np.array(image.shape), name
