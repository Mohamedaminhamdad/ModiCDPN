from __future__ import division
import argparse
import glob
import os
import json
from unicodedata import category
import numpy as np
import scipy.optimize
import numpy as np

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou



def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 
def save_coco_Yolo(args,actual_path):
    with open(args.coco_path) as file:
        coco = json.load(file)
        images=coco['images']
        annotation=coco['annotations']
        category=coco['categories']
    for image in images:
        file_name=image['file_name']
        image_id=image['id']
        width=image['width']
        height=image['height']
        bbox_name='output/'+file_name.replace("png", "txt")
        value=filter(lambda item1: item1['image_id']==image_id,coco['annotations'])
        bbox_yolo=[]
        obj_yolo=[]
        bbox_gt_lis=[]
        annotation_id_list=[]
        obj_gt_list=[]
        bbox_json_list=[]
        with open(bbox_name, 'r') as txt_file:
            for line in txt_file:
                obj,x_rel,y_rel,w_rel,h_rel=line[:-2].split(' ')
                bbox=[float(x_rel)*width-float(w_rel)*width/2,float(y_rel)*height-float(h_rel)*height/2,float(x_rel)*width+float(w_rel)*width/2,float(y_rel)*height+float(h_rel)*height/2]
                bbox_json=[float(x_rel)*width-float(w_rel)*width/2,float(y_rel)*height-float(h_rel)*height/2,float(w_rel)*width,float(h_rel)*height]
                obj=int(obj)+1
                bbox_yolo.append(bbox)
                bbox_json_list.append(bbox_json)
                obj_yolo.append(obj)
            bbox_yolo=np.array(bbox_yolo)
            obj_yolo=np.array(obj_yolo)
            for item2 in value: 
                annotation_id=item2["id"]
                bbox_gt=item2["bbox"] 
                bbox_gt=[bbox_gt[0],bbox_gt[1],bbox_gt[0]+bbox_gt[2],bbox_gt[1]+bbox_gt[3]]
                obj_gt=item2["category_id"]
                bbox_gt_lis.append(bbox_gt)
                obj_gt_list.append(obj_gt)
                annotation_id_list.append(annotation_id)
            bbox_gt=np.array(bbox_gt_lis).reshape(-1,4)
            bbox_yolo=np.array(bbox_yolo).reshape(-1,4)
            bbox_json_list=np.array(bbox_json_list).reshape(-1,4)
            #annotation_id_list=np.array(annotation_id_list)
            idx_gt_actual, idx_pred_actual, ious_actual, label =match_bboxes(bbox_gt, bbox_yolo, IOU_THRESH=0.5)
            new_bbox=bbox_json_list[idx_pred_actual,:]
            for i,annot in enumerate(annotation_id_list):
                if i<len(new_bbox):
                    annotation[annot]["bbox"]=new_bbox[i,:].tolist()
    dic={'images':images,"annotations":annotation,"categories":category}
    with open('keyframe_yolo.json','w') as file:
        json.dump(dic,file)



    

    

if __name__ == '__main__':
    actual_path=os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='keyframe.json', help='*.coco path')
    args = parser.parse_args()
    save_coco_Yolo(args,actual_path)