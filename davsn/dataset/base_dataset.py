from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils import data
import random
import imageio

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_image_crop(self, file):
        img = Image.open(file)
        img = img.convert('RGB')
        img_h = np.array(img).shape[0]
        img_w = np.array(img).shape[1]
        rand = random.randint(1,5)/100
        crop_h = int(np.array(img).shape[0]*rand)
        crop_w = int(np.array(img).shape[1]*rand)
        img = img.crop((crop_w,crop_h,img_w-crop_w,img_h-crop_h))
        # np.array(img).shape
        img = img.resize(self.image_size, Image.BICUBIC)
        return np.asarray(img, np.float32)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def get_labels_sf(self, file):
        img = Image.open(file)
        img = img.resize(self.labels_size, Image.NEAREST)
        return np.asarray(img, np.float32)[:,:,0]

    def get_labels_synthia_seq(self, file):
        # img = Image.open(file)
        lbl = imageio.imread(file, format='PNG-FI')[:, :, 0]
        img = Image.fromarray(lbl)
        img = img.resize(self.labels_size, Image.NEAREST)
        return np.asarray(img, np.float32)

def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
