import torch.utils.data as data
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
import utils.logger as logger
import pickle
from glob import glob 
import random 
import json
from torch.utils.data import DataLoader
import ref
from utils.img import zoom_in
import tqdm
from utils.transform3d import prj_vtx_cam
from scipy.stats import truncnorm 
from transforms3d.quaternions import  quat2mat

class DATASET():
    def __init__(self, arg,split='train',strategy='quat'):
        logger.info('==> initializing {} {} data.'.format(arg.dataset.name, split))
        self.arg=arg
        self.split=split
        self.strategy=strategy
        if arg.dataset.name=='Linemode':
            if split == 'train':
                self.dir=ref.dataset_dir
                with open(os.path.join(ref.dataset_dir,'Linemod_Occluded_train.json'),'r') as f: 
                    self.data=json.load(f)
                self.real_num = len(self.data['annotations'])
            else: 
                self.dir=ref.dataset_dir
                with open(os.path.join(ref.dataset_dir,'Linemod_Occluded_test.json'),'r') as f: 
                    self.data=json.load(f)
                self.real_num = len(self.data['annotations'])
        else: 
            if split == 'train':
                self.dir=ref.dataset_dir
                if arg.train.rot_rep=='allo':
                    with open(os.path.join(ref.dataset_dir,'train_synt_alo.json'),'r') as f:  # Train network allocentricly 
                        self.data=json.load(f)
                else: 
                    with open(os.path.join(ref.dataset_dir,'train_synt.json'),'r') as f: # Train network egocentrilcy 
                        self.data=json.load(f)
                self.real_num = len(self.data['annotations'])
            else:                
                self.dir=ref.dataset_dir
                if arg.pytorch.bbox=='yolo':
                    with open(os.path.join(ref.dataset_dir,'keyframe_yolo.json'),'r') as f:  # Use bbox predicted from YOLOV4-CSP to evaluate network 
                        self.data=json.load(f)
                else:
                    with open(os.path.join(ref.dataset_dir,'keyframe.json'),'r') as f: # Use gt bbox to evaluate network
                        self.data=json.load(f)
                self.real_num = len(self.data['annotations'])

    def load_obj(self, idx):
        return np.array([self.annot[idx]['id']])
    def d_scaled(self, depth, s_box, res):
        """
        compute scaled depth
        """
        r = float(res) / s_box
        return depth / r
    def load_rgb(self, idx):
        return cv2.imread(self.annot[idx]['rgb_pth'])  
    def c_rel_delta(self,c_obj,c_box,wh_box):
        c_delta=np.asarray(c_obj)-np.asarray(c_box)
        c_delta/=np.asarray(wh_box)
        return c_delta 
    def classes(self,idx):
        return self.annot[idx]['name']
    def xywh_to_cs_dzi(self, xywh, s_ratio, s_max=None, tp='uniform'):
        x= xywh[0]
        y=xywh[1]
        w=xywh[2]
        h =xywh[3]
        if tp == 'gaussian':
            sigma = 1
            shift = truncnorm.rvs(-self.arg.augment.shift_ratio / sigma, self.arg.augment.shift_ratio / sigma, scale=sigma, size=2)
            scale = 1+truncnorm.rvs(-self.arg.augment.scale_ratio / sigma, self.arg.augment.scale_ratio / sigma, scale=sigma, size=1)
        elif tp == 'uniform':
            scale = 1+self.arg.augment.scale_ratio * (2*np.random.random_sample()-1)
            shift = self.arg.augment.shift_ratio * (2*np.random.random_sample(2)-1)
        else:
            raise
        c = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])]) # [c_w, c_h]
        s = max(w, h)*s_ratio*scale
        if s_max != None:
            s = min(s, s_max)
        return c, s
    def __getitem__(self, idx):
        if self.split=="train":
            item_={}
            annot=self.data['annotations'][idx]
            obj_id=annot['category_id']
            relative_pose=np.array(annot['relative_pose']["position"]).reshape(-1,3).flatten()
            if self.strategy=='quat':
                relative_quat=np.array(annot['relative_pose']["quaternions"]).reshape(-1,4).flatten()
            else: 
                relative_quat=np.array(annot['relative_pose']["quaternions"]).reshape(-1,4).flatten()
                relative_quat=quat2mat(relative_quat)
            bbox=np.array(annot['bbox']).reshape(-1,4).flatten()
            image_id=annot['image_id']
            value=filter(lambda item1: item1['id']==image_id,self.data['images'])
            for val in value:
                width=val['width']
                height=val['height']
                intrinsics=val['intrinsic']
                image_name=val['file_name']
                if image_name[:5]=='/home':
                    image_name=image_name[-25:]
                if self.arg.dataset.name=='Linemode':
                    rgb_path=os.path.join(ref.dataset_dir,'train',val['file_name'])
                else: 
                    rgb_path=os.path.join(ref.dataset_dir,'images',val['file_name'])
            rgb= cv2.imread(rgb_path)
            c_obj,_=prj_vtx_cam(relative_pose,intrinsics)
            s_max=max(width,height)
            c, s = self.xywh_to_cs_dzi(bbox, self.arg.augment.pad_ratio, s_max,self.arg.dataiter.tp)
            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, 256)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            c = np.array([c_w_, c_h_])
            c_delta=self.c_rel_delta(c_obj,c,np.array([bbox[2],bbox[3]]))
            s = s_
            d_local = self.d_scaled(relative_pose[2], s, self.arg.dataiter.out_res)
            relative_pose = np.append(c_delta, [d_local], axis=0)
            return  obj_id, rgb, np.array(relative_pose).reshape(-1,3), relative_quat, np.array(c), np.array(s),np.array(bbox).reshape(-1,4)
        if self.split=="test":
            item_={}
            annot=self.data['annotations'][idx]
            obj_id=annot['category_id']
            value_name=filter(lambda item1: item1['id']==obj_id,self.data['categories'])
            for val in value_name:
                obj_id=val['name']
            relative_pose=np.array(annot['relative_pose']["position"]).reshape(-1,3).flatten()
            if self.strategy=='quat':
                relative_quat=np.array(annot['relative_pose']["quaternions"]).reshape(-1,4).flatten()
            else: 
                relative_quat=np.array(annot['relative_pose']["quaternions"]).reshape(-1,4).flatten()
                relative_quat=quat2mat(relative_quat)
            bbox=np.array(annot['bbox']).reshape(-1,4).flatten()
            image_id=annot['image_id']
            value=filter(lambda item1: item1['id']==image_id,self.data['images'])
            for val in value:
                width=val['width']
                height=val['height']
                intrinsics=val['intrinsic']
                image_name=val['file_name']
                if self.arg.dataset.name=='Linemode':
                    rgb_path=os.path.join(ref.dataset_dir,'train',val['file_name'])
                else: 
                    rgb_path=os.path.join(ref.dataset_dir,'images',val['file_name'])
            rgb= cv2.imread(rgb_path)
            c_obj,_=prj_vtx_cam(relative_pose,intrinsics)
            s_max=max(width,height)
            c, s = self.xywh_to_cs_dzi(bbox, self.arg.augment.pad_ratio, s_max,self.arg.dataiter.tp)
            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, 256)
            #cv2.imwrite(str(idx)+".png",rgb)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            c = np.array([c_w_, c_h_])
            c_delta=self.c_rel_delta(c_obj,c,np.array([bbox[2],bbox[3]]))
            s = s_
            d_local = self.d_scaled(relative_pose[2], s, self.arg.dataiter.out_res)
            #relative_pose = np.append(c_delta, [d_local], axis=0)
            intrinsic=np.array(intrinsics).reshape(3,3)
            return obj_id, rgb, np.array(relative_pose).reshape(-1,3), relative_quat, np.array(c), np.array(s),np.array(bbox).reshape(-1,4),np.array(intrinsic).reshape(-1,3)
    
    def __len__(self):
        return self.real_num
