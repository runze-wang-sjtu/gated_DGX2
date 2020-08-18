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

Config = Config(sys.argv[1])

class Local_norm(transforms.Normalize):

    def __init__(self):

        self.mask = np.zeros(shape=Config.IMG_SHAPES)
        x0, y0 = int(0.5*self.mask.shape[0]), int(0.5*self.mask.shape[1])
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if (i-x0)**2 + (j-y0)**2 < x0**2:
                    self.mask[i,j] = 1
                else:
                    self.mask[i,j] = 0

    def __call__(self, tensor):

        tensor_arr = tensor.squeeze().numpy()
        mean = self.mean(tensor_arr)
        std = self.std(tensor_arr, mean)
        array = np.zeros(shape=Config.IMG_SHAPES)
        for i in range(tensor_arr.shape[0]):
            for j in range(tensor_arr.shape[1]):
                if self.mask[i, j] == 1:
                    array[i,j] = (tensor_arr[i,j]-mean)/std
        tensor = (torch.from_numpy(array.astype(np.float32))).unsqueeze(0)

        return tensor, mean, std

    def mean(self, array):

        num = 0
        sum = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if self.mask[i,j] == 1:
                    num += 1
                    sum += array[i,j]

        return sum/num

    def std(self,array, mean):

        num = 0
        sum = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if self.mask[i,j] == 1:
                    num += 1
                    sum += (array[i,j] - mean)**2

        return pow(sum/num, 0.5)

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
        self.transforms_oprs["norm"] = transforms.Normalize(mean=[Config.MEAN],std=[Config.STD])
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
