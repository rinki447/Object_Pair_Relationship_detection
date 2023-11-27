import glob
import json
import torch
import os
import numpy as np
import cv2
np.set_printoptions(precision=4)
import copy
import datetime
import time
from torchvision.transforms import transforms, InterpolationMode
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F

BATCHNORM_MOMENTUM = 0.01

#from lib.config import Config

from dataloader.AG_RELATION import AG_relations
from dataloader.action_genome import AG, cuda_collate_fn
from dataloader.base_dataset import BaseDatasetFrames
from dataloader.coco import COCO
from lib.tempura import TEMPURA
#from lib.ds_track import get_sequence
#from lib.evaluation_recall import BasicSceneGraphEvaluator
#from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes

from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
#from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
#from fasterRCNN.lib.model.roi_layers import nms
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


##########################################################################################################################################

''' copying config.py file here   '''
class Config(object):
    def __init__(self):
        """Defaults"""
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.input_dir=None
        self.original_video_path=None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)
        
        if self.mem_feat_lambda is not None:
            self.mem_feat_lambda = float(self.mem_feat_lambda)
        
        
        if self.rel_mem_compute == 'None' :
            self.rel_mem_compute = None
        if self.obj_loss_weighting == 'None':
            self.obj_loss_weighting = None
        if self.rel_loss_weighting == 'None':
            self.rel_loss_weighting = None

    def setup_parser(self):
        """Sets up an argument parser:return:"""
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', default=None, type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='data/ag/', type=str)
        parser.add_argument('-input_dir', default=None, type=str)
        parser.add_argument('-original_video_path', default=None, type=str)
        
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)

        #logging arguments
        parser.add_argument('-log_iter', default=100, type=int)
        parser.add_argument('-no_logging', action='store_true')
 
        # heads arguments
        parser.add_argument('-obj_head', default='gmm', type=str, help='classification head type')
        parser.add_argument('-rel_head', default='gmm', type=str, help='classification head type')
        parser.add_argument('-K', default=4, type=int, help='number of mixture models')

        # tracking arguments
        parser.add_argument('-tracking', action='store_true')

        # memory arguments
        parser.add_argument('-rel_mem_compute', default=None, type=str, help='compute relation memory hallucination [seperate/joint/None]')
        parser.add_argument('-obj_mem_compute', action='store_true')
        parser.add_argument('-take_obj_mem_feat', action='store_true')
        parser.add_argument('-obj_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-rel_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-mem_fusion',default='early', type=str, help='early/late')
        parser.add_argument('-mem_feat_selection',default='manual', type=str, help='manual/automated')
        parser.add_argument('-mem_feat_lambda',default=None, type=str, help='selection lambda')
        parser.add_argument('-pseudo_thresh', default=7, type=int, help='pseudo label threshold')

        # uncertainty arguments
        parser.add_argument('-obj_unc', action='store_true')
        parser.add_argument('-rel_unc', action='store_true')

        #loss arguments
        parser.add_argument('-obj_loss_weighting',default=None, type=str, help='ep/al/None')
        parser.add_argument('-rel_loss_weighting',default=None, type=str, help='ep/al/None')
        parser.add_argument('-mlm', action='store_true')
        parser.add_argument('-eos_coef',default=1,type=float,help='background class scaling in ce or nll loss')
        parser.add_argument('-obj_con_loss', default=None, type=str,  help='intra video visual consistency loss for objects (euc_con/info_nce)')
        parser.add_argument('-lambda_con', default=1,type=float,help='visual consistency loss coef')
        return parser


############################################### gt_annotation   #######################################################################################
class annotation():
    def gt_annotation(saved_json_path,key_vid,entry,frame_box):

        with open(saved_json_path,'r') as f: 
            json_file= json.load(f)    

        frame_list=list(json_file[f"{key_vid}"].keys())
        frame_name_list=np.unique(frame_box[:,0])
        uniq_frames=np.unique(entry["boxes"][:,0])
       
        gt_annotation_video = []
        entry_frame={}
            
        for i in uniq_frames:####### each frame j

            ind=torch.nonzero(entry["boxes"][:,0]==i).view(1,-1)
            ind=ind.squeeze(0)
            
            entry_frame["boxes"]=entry["boxes"][ind]
            entry_frame["pred_scores"]= entry["pred_scores"][ind]
            entry_frame["distribution"]= entry["distribution"][ind]
            entry_frame["labels"]= entry["labels"][ind]

            indice=np.arange(entry_frame["boxes"].shape[0])
            h_idx=np.where(entry_frame["labels"]==0)


            if len(h_idx[0]) > 0: # if there is atleast one human box
                entry_hum=entry_frame["distribution"][h_idx[0]]
                human_idx_local = torch.argmax(entry_hum[:,0]) 
                human_idx=indice[h_idx][human_idx_local]

            else:

                human_idx = torch.argmax(entry_frame['distribution'][:,0])  # the local bbox index with highest human score in this frame

            human_idx=human_idx.item() 
            human_bbox=entry_frame["boxes"][human_idx,1:].cpu().numpy().tolist()
            human=[]
            human.append(human_bbox)

            obj_box_len=entry_frame["boxes"][:,0].shape[0]-1 # 1 of the boxes belongs to human
            box_idx=np.arange(0,(entry_frame["boxes"].shape[0])) #index of all boxes
            obj_box_idx = np.delete(box_idx, np.where(box_idx == int(human_idx))) # index of non-human boxes
                            
            gt_annotation_frame = [{'person_bbox':np.array(human, dtype=np.float32), "frame": f"{i}"}] 

            obj_cls_list=entry_frame["labels"][obj_box_idx]
            c=0
            for k in obj_cls_list: 
                    
                    aa={}
                    aa['class'] = int(k)
                    id=obj_box_idx[c]
                    aa['bbox'] = np.array(entry_frame['boxes'][id,1:].cpu().numpy().tolist())                      
                    gt_annotation_frame.append(aa)
                    c=c+1
                   
            gt_annotation_video.append(gt_annotation_frame) ##### frame wise append
                        
        return(gt_annotation_video)

############################################ load data saved using object_detector.py #######################################################
class load_saved_detection():
  def load_entry(saved_json_path,vid,original_video_path,fmap_path,directory_path):
    entry={}
    boxes=[]
    distribution=[]
    pred_scores=[]
    pred_labels=[]
    frame_box=[]
    frames=[]
    im_scales=[]

    # open json file, saved by object_saver.py, corresponding to each video
    with open(saved_json_path,'r') as f: 
      json_file= json.load(f)

    key_vid=vid     
    frame_list=list(json_file[f"{key_vid}"].keys())

    #*** start: extra step for image and im_scale (if not saved by object_saver.py)********
    
    # get original file path of the same video from text file, as saved by annotation.py
    with open(original_video_path,'r') as f: 
      video_path=f.read().splitlines() # save all video path
    for pth in video_path:
      if pth.split("/")[-1]==key_vid: 
        my_video_path=pth # save single (matched one) video path

    c=0
    for i,key_frame in enumerate(frame_list):

      frame_path=os.path.join(my_video_path,f"{key_frame}.jpg")
      image=cv2.imread(frame_path,cv2.IMREAD_UNCHANGED)
      im, im_scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 384, 384) #adopted from AG_dataset of action_genome.py
      im_scales.append(im_scale)
      frames.append(im)
      
      #**************************** ends extra steps for image and im scale ***************

      ################## collecting saved box data for single video into entry dictionary #####################
      
      box_list=list(json_file[f"{key_vid}"][f"{key_frame}"].keys())

      for key_box in box_list: ###### for all boxes in single frame
        frame_box.append([key_frame,int(i),int(c)]) #frame_name, frame_number,box_global_number
      
        box_coord=(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["box_detail"]) # "box_detail" keyword is used while saving
        box_coord.insert(0,int(i)) # frame number should be the first column in entry["boxes"] (this is not the original one saved during object detection)
        boxes.append(box_coord) # all boxes for all frames in single video are being saved in "boxes" list

        distribution.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["distribution"])
        pred_labels.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["pred_label"])
        pred_scores.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["pred_score"])

        c=c+1

    
    blob = im_list_to_blob(frames)     #frames= list of all images
    im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)  
    im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)

    # info about all boxes from all frames of single video are saved in "entry"
    entry["boxes"]=torch.tensor(boxes)    
    entry["distribution"]=torch.tensor(distribution)  
    entry["labels"]=torch.tensor(pred_labels) 
    entry["pred_scores"]=torch.tensor(pred_scores)  
    entry["im_info"]=im_info

    # upload f_map for each frame
    f_fmap=fmap_path
    npz_file= np.load(f_fmap)
    entry["fmaps"]=torch.tensor(npz_file["data_fmap"]) # key to load data here is "data_fmap" as used while saved
  
    # upload features for each box in a frame and likewise for all frame
    features=[]
    for frame in frame_list:
              
      f=os.path.join(directory_path,f'{vid}/{frame}/npz_file.npz') 
      npz_file= np.load(f)
      npz_length=len(npz_file)

      for i in range(npz_length-1): # "allow pickle" is also in the list at the end, omit that
        features.append(npz_file[f"box:no:{i}"]) #### that is how loading custom key was saved
    
    entry["features"]=torch.tensor(np.array(features))

    frame_box=np.array(frame_box) # list of frame_name, frame_number,box_global_number for all box in a video
    return entry,frame_box



############################# filter_no_human.py ###############################################

class filter_nonhuman():
  def filter(entry, frame_box):

    uniq_frames=np.unique(entry["boxes"][:,0])
    frame_id=[]
    index=torch.tensor([])
    entry_frame={}
    ent={}
    frame_no=0

    # find human/"person" boxes in each frame
    for i in uniq_frames:
    
      ind=torch.nonzero(entry["boxes"][:,0]==i).view(1,-1)
      indice=ind.squeeze(0)
      ent["labels"]= entry["labels"][indice]


      if 0 in ent["labels"]: # if there is atleast 1 "person" box in a frame, the frame is kept
        frame_id.append(frame_no)
        index=torch.cat((index,indice),0) # global index of that frame_boxes are saved
      
      frame_no=frame_no+1

    index=index.squeeze(0).long() # list of global index of all boxes (belonging to filtered human frames) of a video 
    
    # box, score,distribution, label are changed for all boxes
    
    entry_frame["boxes"]=entry["boxes"][index]
    entry_frame["pred_scores"]= entry["pred_scores"][index]
    entry_frame["distribution"]= entry["distribution"][index]
    entry_frame["labels"]= entry["labels"][index] 
    entry_frame["features"]= entry["features"][index] 

    frame_id=torch.tensor(frame_id).long() # list of indices of all filtered human frames of a video 

    # fmap and im_info are changed as per frame basis

    entry_frame["fmaps"]=entry["fmaps"][frame_id] 
    entry_frame["im_info"]=entry["im_info"][frame_id]

    frame_box1=frame_box[index.cpu().numpy()] ### filtered list of (frame_name,frame_no,global_box_no)

    return entry_frame, frame_box1 


############################### non_max_suppression.py ######################################
class non_max():
  def nms(entry):
    nms_box=torch.tensor([])
    nms_scores=torch.tensor([])
    nms_distribution=torch.tensor([])
    nms_label=torch.tensor([])
    nms_feature=torch.tensor([])
    uniq_frames=np.unique(entry["boxes"][:,0].cpu().numpy())
    entry_idx=torch.tensor([])
    entry_idx = entry_idx.to(dtype=torch.long)
    idx1=torch.tensor(np.arange(0,entry["boxes"].shape[0]))

    sum=0
  
    for i in uniq_frames:

      idx2=idx1[entry["boxes"][:,0]==i].clone().detach()

      frame_box=entry["boxes"][entry["boxes"][:,0]==i]
      frame_scores=entry["pred_scores"][entry["boxes"][:,0]==i]
      frame_distribution=entry["distribution"][entry["boxes"][:,0]==i]
      frame_labels=entry["labels"][entry["boxes"][:,0]==i]
      frame_features=entry["features"][entry["boxes"][:,0]==i]
      uniq_labels=np.sort(np.unique(frame_labels))

      for j in uniq_labels:
        idx3=idx2[frame_labels==j].clone().detach()
      
        class_box=frame_box[frame_labels==j]
        cls_box_list=class_box.cpu().numpy().tolist()
        class_scores=frame_scores[frame_labels==j]
        cls_scores_list=class_scores.cpu().numpy().tolist()
        class_distribution=frame_distribution[frame_labels==j]
        class_label=frame_labels[frame_labels==j]
        class_feature=frame_features[frame_labels==j]


        boxes_for_nms = [[x1, y1, x2, y2] for _,x1, y1, x2, y2 in cls_box_list]
        confidences = [confidence for confidence in cls_scores_list]

        keep = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, 0.0, 0.5)

        

        nms_box=torch.cat((nms_box,class_box[keep]),0)
        nms_scores=torch.cat((nms_scores,class_scores[keep]),0)
        nms_distribution=torch.cat((nms_distribution,class_distribution[keep]),0)
        nms_label=torch.cat((nms_label,class_label[keep]),0)
        nms_feature=torch.cat((nms_feature,class_feature[keep]),0)
        idx4=idx3[keep].clone().detach().long()
      
        entry_idx=torch.cat((entry_idx,idx4),0)
      

    entry["boxes"]=nms_box
    entry["pred_scores"]=nms_scores
    entry["distribution"]=nms_distribution
    entry["labels"]=nms_label
    entry["features"]=nms_feature

    print("total box after nms",entry["boxes"].shape[0])

    return entry,entry_idx       

############################## pair.py #####################################################

class pair_maker(nn.Module):

    def __init__(self, train, object_classes, use_SUPPLY, mode='predcls'):
        super(pair_maker, self).__init__()

        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode
        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_coco.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self,gt_annotation,entry):

            im_info=entry["im_info"]
            # how many bboxes we have
            bbox_num = 0
            im_idx = []  # which frame are the relations belong to
            pair = []

            for i in gt_annotation:
                bbox_num += len(i)
                
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 0#?????????????????????????????
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        bbox_idx += 1
            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

            counter = 0
            FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)
            
            #use entry["fmaps"] for each video as FINAL_BASE_FEATURES
            FINAL_BASE_FEATURES =entry["fmaps"]
            ###################################################################################    
          
            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
            FINAL_BBOXES = copy.deepcopy(FINAL_BBOXES.cuda(0))
            FINAL_BASE_FEATURES = copy.deepcopy(FINAL_BASE_FEATURES.cuda(0))
            FINAL_FEATURES=entry["features"].cuda(0)

            if self.mode == 'predcls':

                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                          }

            return entry


###########################################################################################################################################

def main(conf):
    #conf = Config()
    conf.datasize="large"
    gpu_device = torch.device('cuda:0')
     
    data_path=conf.data_path ######################## used for AG and coco only
    
    AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True)
    AG_rel_classes=AG_relations(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True)  ##### rpossible elations classes taken from AG
    COCO_dataset = COCO(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True)   #### obj_classes taken from coco
    
    
    ##### collecte video names from data generated by object_saver.py
    directory_path = conf.input_dir ######## change for new dataset, for train/test video case
    
    all_v_idx=next(os.walk(directory_path))[1]
    
    gpu_device = torch.device('cuda:0')
    
    pairing=pair_maker(train=False,object_classes=COCO_dataset.object_classes, use_SUPPLY=True, mode='predcls').to(device=gpu_device)
    
    
    model = TEMPURA(mode="predcls",
                   attention_class_num=len(AG_dataset.attention_relationships),
                   spatial_class_num=len(AG_dataset.spatial_relationships),
                   contact_class_num=len(AG_dataset.contacting_relationships),
                   obj_classes=COCO_dataset.object_classes,
                   enc_layer_num=conf.enc_layer,
                   dec_layer_num=conf.dec_layer,
                   obj_mem_compute = conf.obj_mem_compute,
                   rel_mem_compute = conf.rel_mem_compute,
                   take_obj_mem_feat= conf.take_obj_mem_feat,
                   mem_fusion= conf.mem_fusion,
                   selection = conf.mem_feat_selection,
                   selection_lambda=conf.mem_feat_lambda,
                   obj_head = conf.obj_head,
                   rel_head = conf.rel_head,
                   K = conf.K,
                   tracking= conf.tracking).to(device=gpu_device)
    
    model.eval()
     
    
    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    ckpt_clone = ckpt['state_dict'].copy()
    
    
    for k in list(ckpt['state_dict'].keys()):
      if 'object_classifier' in k or 'obj_embed' in k:
    
        ckpt_clone.pop(k)
    
    model.load_state_dict(ckpt_clone, strict=False) 
    
    print('*'*50)
    print('CKPT {} is loaded'.format(conf.model_path))
    
    count=0
    
    for vid in list(all_v_idx) : 
    
    
      saved_json_path=os.path.join(directory_path,f"{vid}/Activity_BBox.json")   
      saved_fmap_path=os.path.join(directory_path,f"{vid}/npz_fmap.npz")  
      original_video_path=conf.original_video_path
      ####### load saved entry boxes, labels, distributions, score, fmap, im_info #####
    
      #frame_box1=[frame_name, frame_number,box_global_number] for all box
      entry,frame_box1=load_saved_detection.load_entry(saved_json_path,vid,original_video_path,saved_fmap_path,directory_path) 
      unq_f1=np.unique(entry["boxes"][:,0].cpu().numpy())
      print("unq_fr 1 length",len(unq_f1))
      #print(frame_box1)
      
    
      ######################  filter non human frames #################
    
      # frame_box1= list of global index of the boxes filtered
      entry, frame_box11=filter_nonhuman.filter(entry, frame_box1)
    
    
      unq_f2=np.unique(entry["boxes"][:,0].cpu().numpy())
      print("unq_fr 2 length",len(unq_f2))
      #print(frame_box11)
      
      
      ################ apply nms ##################################
     
      entry,entry_idx=non_max.nms(entry)
       
      frame_box1=frame_box11[entry_idx.cpu().numpy()]
    
      unq_f3=np.unique(entry["boxes"][:,0].cpu().numpy())
      print("unq_fr 3 length",len(unq_f3))
      print("bbox length",len(frame_box1[:,0]))
      print("feature length",entry["features"].shape)
     
      # im_info and fmap are related to each image, not boxes of the image, so nms does not change them
            
      ############################### create gt_annotation ###########################################################
      
      gt_annotation_video=annotation.gt_annotation(saved_json_path,vid,entry,frame_box1)
      gt_annotations=gt_annotation_video
    
      bbox_num=0
      for i in gt_annotations:
        bbox_num += len(i)
      print(bbox_num)
      
    
      ############################### saving all frame names and paths in a video #######################################################
    
    
      with open(original_video_path,'r') as f: ##### change for dataset, train/test case
        video_path=f.read().splitlines() ####save all video path
      for i,pth in enumerate(video_path):
        if pth.split("/")[-1]==vid: 
          my_video_path=pth 
    
      with open(saved_json_path,'r') as f:  #####change for dataset, train/test video case
                json_file= json.load(f)
    
      ###### getting frame names ##################        
      
      frame_list=np.unique(frame_box1[:,0])
      #print("frame 2 list", frame_list)
    
      
      frame_paths=[]
      for kk in frame_list:
        
        frame_paths.append(os.path.join(my_video_path, f'{kk}.jpg')) 
      
      ##### human_object pairing (following predcls part of object_detector.py)#####################################
    
      im_info=entry["im_info"]
      entry=pairing(gt_annotations, entry)
    
      #if conf.tracking:
          #get_sequence(entry, gt_annotations, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
    
    
      ####### relationship prediction for predcls tempure test mode ###################################################
    
    
      pred = model(entry,phase='test', unc=False) 
     
      ############################## print objects and relations per frame ###########################################               
      row=[]
      prev_row_n=0
      a_rel_frame=[]
      for i,key_frame in enumerate(frame_list):
        all_box_idx=torch.nonzero(entry["boxes"][:,0]==i).view(-1)  
        #print(entry["pred_labels"][all_box_idx])
        #print("all_box_idx test print",all_box_idx)
        
        object_box_n=len(all_box_idx)-1
        hum_box_id=pred["human_idx"][i]
        
        if row==[]:
          prev_row_n=0
        else:
          prev_row_n=np.sum(row)
    
        row.append(object_box_n)
        end_row_n=np.sum(row)
    
        
        a_rel_frame=[]
        c_rel_frame=[]
        s_rel_frame=[]
    
        frame_path=os.path.join(my_video_path,f"{key_frame}.jpg")
    
        print(f"frame_path:{frame_path}")
        
    
        for j in range(prev_row_n,end_row_n): ## all boxes in single frame
          
          a_rel_class=[]
          #hum=entry["pred_labels"][entry["pair_idx"][j,0]]
          hum_class=COCO_dataset.object_classes[1] ##### all human labels are marked as "80"
          obj=pred["pred_labels"][pred["pair_idx"][j,1]]
          obj_class=COCO_dataset.object_classes[obj+1]
        
          values_a = pred["attention_distribution"][j, :].detach().cpu().numpy()
          top_one_indices_a = list(np.argsort(values_a)[::-1][:1])
          elements_a = [AG_rel_classes.attention_relationships[i] for i in top_one_indices_a]
               
    
          values_c = pred["contacting_distribution"][j, :].detach().cpu().numpy()
          top_three_indices_c = np.argsort(values_c)[::-1][:3]
          elements_c = [AG_rel_classes.contacting_relationships[i] for i in top_three_indices_c]
    
          values_s = pred["spatial_distribution"][j, :].detach().cpu().numpy()
          top_three_indices_s = np.argsort(values_s)[::-1][:3]
          elements_s = [AG_rel_classes.spatial_relationships[i] for i in top_three_indices_s]
         
          
          print(f"caption:{hum_class},{elements_a}/{elements_c}/{elements_s},{obj_class}")
    
    
        print("done for one frame")
        exit()
             
    #total_time = time.time() - start_time
    #total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #print('Inference time {}'.format(total_time_str), flush=True)
    # if conf.output_dir is not None:
    #     with open(conf.output_dir+"log_"+conf.mode+".txt", "a") as f:
    #                 f.truncate(0)
    #                 f.close()
    '''constraint_type = 'with constraint'
    print('-'*10+constraint_type+'-'*10)
    evaluator1.print_stats(log_file=log_val)
    
    constraint_type = 'semi constraint'
    print('-'*10+constraint_type+'-'*10)
    evaluator2.print_stats(log_file=log_val)
    
    constraint_type = 'no constraint'
    print('-'*10+constraint_type+'-'*10)
    evaluator3.print_stats(log_file=log_val)'''



if __name__ == '__main__':
    print("###########################")
    print("Calling main()")
    print("###########################")
    conf = Config()
    print("Calling config()")
    print("###########################")
    main(conf)


