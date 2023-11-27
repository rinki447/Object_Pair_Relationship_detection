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

class AG_relations(Dataset):

    def __init__(self, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


  


