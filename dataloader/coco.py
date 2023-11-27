import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
#from scipy.misc import imread
from PIL import Image
import numpy as np
import pickle
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from cv2 import imread
import cv2

class COCO(Dataset):

    def __init__(self, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):

        root_path = data_path
        

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes_coco.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        