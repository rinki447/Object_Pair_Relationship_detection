import numpy as np

import torch

from PIL import Image

from PIL import ImageFile

from torch.utils.data import Dataset

import glob
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
import cv2

 

class BaseDatasetFrames(Dataset):

    def __init__(self, data_file_path,transform,max_frames=256):

        super().__init__()

        self.transform = transform

        self.max_frames = max_frames

        self.data = []

        self.vid_ids = []

        with open(data_file_path,'r') as f:

            data_path = f.read().splitlines()

       

        for j,path in enumerate(data_path):

            frame_paths = sorted(glob.glob(path+'/*jpg'))

            tot_frames = len(frame_paths)

            self.vid_ids.append(path.split('/')[-1])
            frames,frame_ids =self._sample_frames( frame_paths,tot_frames)
            self.data.append((frames,frame_ids))
            

            # if j==2:

                # break

           

 

    def _sample_frames(self, frame_paths,tot_frames):

        max_frames = self.max_frames

        keep_idx = np.arange(0, max_frames ) / max_frames * tot_frames

        keep_idx = np.round(keep_idx).astype(np.int64)

        keep_idx[keep_idx >= tot_frames] = tot_frames - 1

        #print("sampled index",keep_idx)

        sampled_frame_paths = []

        # print(len(keep_idx),max_frames)

        assert len(keep_idx) == max_frames

        unq_keep_idx = np.unique(keep_idx)

        # print(keep_idx)

        # print(tot_frames)

        # print(unq_keep_idx)

        # exit()

       

        for j in unq_keep_idx:

            sampled_frame_paths.append(frame_paths[j])

 

        return sampled_frame_paths, unq_keep_idx

   

    def __len__(self):

        return len(self.data)

   

    def __getitem__(self, index):

        frame_paths,sampled_frame_ids = self.data[index]

        vid_id = self.vid_ids[index]

        #print("video name",vid_id)
        #print("frame name",frame_paths)

        frames = []
        path_f=[]
        #processed_ims = []
        im_scales = []
        #path_list_all
        for i,path in enumerate(frame_paths):
            image=cv2.imread(path,cv2.IMREAD_UNCHANGED)
            #print(image.shape)
            #image = Image.open(path).convert('RGB')  

            #frames.append(self.transform(image))
            #image2=self.transform(image)
            im, im_scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 384, 384) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            #print(im.shape)
            #print(im_scale)

            path_n=path.split('/')[-1] 
            path_f.append(path_n.strip('.jpg'))
            im_scales.append(im_scale)
            frames.append(im)
            if (i==0):
               path_list=path

 

        #frames = torch.stack(frames,dim=0)


        ######## extra part ###############################################################
        blob = im_list_to_blob(frames)     #(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)   # what is im_scales  ?????
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        #######################################################################################
 

        #return frames,vid_id,sampled_frame_ids
        return img_tensor, im_info, gt_boxes, num_boxes, index, vid_id, path_f, path_list