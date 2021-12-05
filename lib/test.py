import torch
import numpy as np
import os
from utils.utils import AverageMeter
from utils.eval import Evaluation
from progress.bar import Bar
import os
import time
from transforms3d.quaternions import quat2mat, qmult
from transforms3d.euler import euler2quat, euler2mat
def allocentric2egocentric(qt, T):
    """
    Transform allocentric quat rotation to egocentric

    Args:
        qt (np.ndarray): predicted quaternion allocentricly 
        T ([np.ndarray): predicted translation

    Returns:
        quat [np.ndarray]: predicted quat egocentric
    """
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(quat, qt)
    return quat
def allocentric2egocentricR(R_pr, T):
    """
    Transform allocentric Rotation matrix  to egocentric

    Args:
        qt (np.ndarray): predicted Rotation matrix allocentricly 
        T ([np.ndarray): predicted translation

    Returns:
        quat [np.ndarray]: predicted Rotation matrix  egocentric
    """
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    Rc = euler2mat(-dy, -dx, 0, axes='sxyz')
    R = np.matmul(Rc, R_pr)
    return R
def test(epoch, cfg, data_loader, model, classes,prj_vtx,model_info):
    """
    Test function 

    Args:
        epoch (epoch): [description]
        cfg (dic): configuration in dic type
        data_loader : test-data loader 
        model: Network model
        classes (dic): dictionary containing classes of YCB-Video dataset
        prj_vtx (dic): 3D-Models (x,y,z)
        model_info (dic): dictionary containing model info such as diameter ...etc.
    """
    model.eval()
    num_iters = len(data_loader)
    Eval = Evaluation(prj_vtx,classes,model_info)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    vis_dir = os.path.join(cfg.pytorch.save_path, 'test_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for i, (obj, rgb, relative_pose, relative_quat, c, s, bbox,intrinsics) in enumerate(data_loader):
        if cfg.pytorch.gpu > -1:
            inp_var = rgb.cuda(cfg.pytorch.gpu, non_blocking=True).float()
            relative_pose=relative_pose.cuda(cfg.pytorch.gpu,non_blocking=True).float().squeeze(1)
            relative_quat=relative_quat.cuda(cfg.pytorch.gpu,non_blocking=True).float().squeeze(1)
            c=c.cuda(cfg.pytorch.gpu,non_blocking=True).float()
            s=s.cuda(cfg.pytorch.gpu,non_blocking=True).float()
            intrinsics=intrinsics.cuda(cfg.pytorch.gpu,non_blocking=True).float()
            bbox=bbox.cuda(cfg.pytorch.gpu,non_blocking=True).float().squeeze(1)
        else:
            inp_var = rgb.float()
            relative_pose=relative_pose.float().squeeze(1)
            relative_quat=relative_quat.float().squeeze(1)
            c=c.float()
            s=s.float()
            intrinsics=intrinsics.float()
            bbox=bbox.float().squeeze(1)
        bs = len(inp_var)
        # forward propagation
        pred_rot, pred_trans = model(inp_var)
    
        ratio_delta_c = pred_trans[:,:2]
        ratio_depth = pred_trans[:,2]
        pred_depth = ratio_depth * (cfg.dataiter.out_res / s)
        pred_c = ratio_delta_c * bbox[:,2:] + c
        pred_x = (pred_c[:,0] - intrinsics[:,0, 2]) * pred_depth / intrinsics[:,0, 0]
        pred_y = (pred_c[:,1] - intrinsics[:,1, 2]) * pred_depth / intrinsics[:,1, 1]
        T_vector_trans = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), pred_depth.unsqueeze(0)),0)
        T_vector_trans1=T_vector_trans.transpose(0,1).cpu()
        
        #T_vector_trans1=pred_trans.cpu()
        pred_rot=pred_rot.cpu()
        intrinsics=intrinsics.cpu()
        relative_quat=relative_quat.cpu()
        relative_pose=relative_pose.cpu()
        
        if cfg.network.rot_representation=='quat':
            for idx in range(bs):
                T_begin = time.time()
                q=pred_rot[idx].detach().numpy()
                if cfg.train.rot_rep=='allo':
                    q=allocentric2egocentric(q,T_vector_trans1[idx].detach().numpy())
                pose_est = np.concatenate((quat2mat(q), np.asarray((T_vector_trans1[idx].detach().numpy()).reshape(3, 1))), axis=1)
                pose_gt = np.concatenate((quat2mat(relative_quat[idx].detach().numpy()), np.asarray((relative_pose[idx].detach().numpy()).reshape(3, 1))), axis=1)
                obj_=obj[idx]
                intrinsics_=intrinsics[idx]
                Eval.quaternion_est_all[obj_].append(quat2mat(q))
                Eval.quaternion_gt_all[obj_].append(quat2mat(relative_quat[idx].detach().numpy()))
                Eval.pose_est_all[obj_].append(pose_est)
                Eval.pose_gt_all[obj_].append(pose_gt)
                Eval.translation_gt[obj_].append(np.asarray((relative_pose[idx].detach().numpy()).reshape(3, 1)))
                Eval.translation_est[obj_].append(np.asarray((T_vector_trans1[idx].detach().numpy()).reshape(3, 1)))
                Eval.num[obj_] += 1
                Eval.camera_k[obj_].append(intrinsics_)
                Eval.numAll += 1
        else: 
            for idx in range(bs):
                R=pred_rot[idx].detach().numpy()
                if cfg.train.rot_rep=='allo':
                    R=allocentric2egocentricR(R,T_vector_trans1[idx].detach().numpy())
                pose_est = np.concatenate((R, np.asarray((T_vector_trans1[idx].detach().numpy()).reshape(3, 1))), axis=1)
                pose_gt = np.concatenate((relative_quat[idx].detach().numpy(), np.asarray((relative_pose[idx].detach().numpy()).reshape(3, 1))), axis=1)
                obj_=obj[idx]
                intrinsics_=intrinsics[idx]
                Eval.quaternion_est_all[obj_].append(R)
                Eval.quaternion_gt_all[obj_].append(relative_quat[idx].detach().numpy())
                Eval.pose_est_all[obj_].append(pose_est)
                Eval.pose_gt_all[obj_].append(pose_gt)
                Eval.translation_gt[obj_].append(np.asarray((relative_pose[idx].detach().numpy()).reshape(3, 1)))
                Eval.translation_est[obj_].append(np.asarray((T_vector_trans1[idx].detach().numpy()).reshape(3, 1)))
                Eval.num[obj_] += 1
                Eval.camera_k[obj_].append(intrinsics_)
                Eval.numAll += 1
        Bar.suffix = 'test Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    Eval.te3d()
    Eval.calculate_class_avg_rotation_error('mohamed/')
    Eval.evaluate_pose()
    Eval.evaluate_pose_add('ADD')
    Eval.evaluate_pose_add('ADD-S','symmetric')
    Eval.evaluate_trans()


    return 

