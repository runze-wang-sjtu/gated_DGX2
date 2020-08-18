import torch
import numpy as np
import sys
from util.config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#import nori2 as nori

"""
Define a base dataset contains some function I always useself.
"""


class Local_norm(transforms.Normalize):

    def __init__(self):
        pass

    def __call__(self, tensor):

        tensor_arr = tensor.squeeze().numpy()
        mean = tensor_arr.mean()
        std = tensor_arr.std()
        for i in range(tensor_arr.shape[0]):
            for j in range(tensor_arr.shape[1]):
                tensor_arr[i,j] = (tensor_arr[i,j]-mean)/std
        tensor = (torch.from_numpy(tensor_arr.astype(np.float32))).unsqueeze(0)

        return tensor, mean, std


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


    def transform_initialize(self, crop_size, config=['resize', 'to_tensor']):
        """
        Initialize the transformation oprs and create transform function for img
        """
        self.transforms_oprs = {}
        self.transforms_oprs["hflip"]= transforms.RandomHorizontalFlip(0.5)
        self.transforms_oprs["vflip"] = transforms.RandomVerticalFlip(0.5)
        self.transforms_oprs["random_crop"] = transforms.RandomCrop(crop_size)
        self.transforms_oprs["to_tensor"] = transforms.ToTensor()
        self.transforms_oprs["norm"] = transforms.Normalize(mean=[0],std=[1])
        self.transforms_oprs["resize"] = transforms.Resize(crop_size)
        self.transforms_oprs["center_crop"] = transforms.CenterCrop(crop_size)
        self.transforms_oprs["rdresizecrop"] = transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0), ratio=(1,1), interpolation=2)
        self.transforms_image = transforms.Compose([self.transforms_oprs[name] for name in config])
        self.transforms_mask = transforms.Compose([transforms.Resize(crop_size),transforms.ToTensor()])

    def loader(self, **args):
        return DataLoader(dataset=self, **args)

    @staticmethod
    def read_img(path):
        """
        Read Images
        """
        img = Image.open(path)

        return img

    # @staticmethod
    # def norm_counter(path):
    #     """
    #     count mean and std of img
    #     """
    #     img = Image.open(path)
    #     arr = np.array(img)
    #     mean, std = arr.mean(), arr.std()
    #
    #     return mean, std

class NoriBaseDataset(BaseDataset):
    """
    Implement for reading nori data which is important in Megvii
    """

    def __init__(self, nori_list_path, nori_path):
        self.nori_list, self.cls_list, self.img_nr = self.initialize_nori(nori_list_path, nori_path)

    def initialize_nori(self, nori_list_path, nori_path):
        nori_list = []
        cls_list = []
        with open(nori_list_path, 'r') as f:
            for line in f:
                terms = line.strip().split('\t')
                assert len(terms) == 3
                nori_id, cls_id, img_id = terms
                nori_list.append(nori_id)
                cls_list.append(cls_id)
        nr = nori.open(nori_path, 'r')
        return nori_list, cls_list, nr
    def __len__(self):
        return len(self.nori_list)

    @staticmethod
    def read_img(nori_id):
        """
        Read Images
        """
        img_bytes = self.img_nr.get(nori_id)
        nparr = np.fromstring(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return np.array(img)[:,:,::-1]

    def __getitem__(self, index):
        return read_img(self.nori_list[index]), self.cls_list[index]
