import numpy as np

from davsn.dataset.base_dataset import BaseDataset
import cv2

class ViperDataSet(BaseDataset):
    def __init__(self, root, list_path, set='train',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        # map to cityscape's ids
        self.id_to_trainid = {3: 0, 4: 1, 9: 2, 11: 3, 13: 4, 14: 5, 7: 6, 8: 6, 6: 7, 2: 8, 20: 9, 24: 10, 27: 11,
                          26: 12, 23: 13, 22: 14}
        self.ignore_ego_vehicle = True

    def get_metadata(self, name):
        img_file = self.root / 'train/img' / name
        label_file = self.root / 'train/cls' / name.replace('jpg','png')
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        if self.ignore_ego_vehicle:
            lbl_car = label == 24
            ret, lbs, stats, centroid = cv2.connectedComponentsWithStats(np.uint8(lbl_car))
            lb_vg = lbs[-1, lbs.shape[1] // 2]
            if lb_vg > 0:
                label[lbs == lb_vg] = 0
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        frame = int(name.split('/')[-1].replace('.jpg','')[-5:])
        name_kf = name.replace(str(frame).zfill(5) + '.jpg', str(frame - 1).zfill(5) + '.jpg')
        file_kf = self.root / 'train/img' / name_kf
        image_kf = self.get_image(file_kf)
        image_kf = self.preprocess(image_kf.copy())
        return image.copy(), label_copy.copy(), image_kf.copy(), np.array(image.shape), name
