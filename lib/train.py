import numpy as np
import os
from utils.utils import AverageMeter
import cv2
from progress.bar import Bar
import os
import utils.logger as logger
import time

def train(epoch, arg, data_loader, model, criterions, optimizer=None):
    """

    Args:
        epoch ( int): epoch
        arg ([dic]): Config dic containing all config params
        data_loader : train Dataloader
        model : Psition estimation network
        criterion (class tuple): Loss functions
        optimizer: choosen optimizer
    return: 
    """
    model.train()
    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(arg.pytorch.exp_id[-60:]), max=num_iters)

    time_monitor = True
    vis_dir = os.path.join(arg.pytorch.save_path, 'train_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    for i, (obj, rgb, relative_pose, relative_quat, c, s,bbox) in enumerate(data_loader):
        cur_iter = i + (epoch - 1) * num_iters
        if arg.pytorch.gpu > -1:         # Check if GPU is used or CPU 
            inp_var = rgb.cuda(arg.pytorch.gpu, non_blocking=True).float() # Pass Image to cuda 
            relative_quat_ = relative_quat.cuda(arg.pytorch.gpu, non_blocking=True).float() # Pass rotation representation to cuda()
            relative_pose_ = relative_pose.cuda(arg.pytorch.gpu, non_blocking=True).float() # Pass translation to cuda()
        else: # If GPU is not used leave image, and its annotation in CPU 
            inp_var = rgb
            relative_quat_ = relative_quat.float()
            relative_pose_ = relative_pose.float()
        if arg.network.rot_representation=='quat': # If quaternion is used squeeze 
            relative_quat_=relative_quat_.squeeze(1) # Change quat shape to [bs,4], for rotation it is left to be [bs,3,3]
        relative_pose_=relative_pose_.squeeze(1) # Change trans shape to [bs,3]
        bs = len(inp_var) # Batch size 
        # forward propagation
        T_begin = time.time()
        pred_rot, pred_trans = model(inp_var) # Output rotation and translation in [bs, 4] for quaternion [bs,3,3] for 6D-Head, and translation [bs,3]
        T_end = time.time() - T_begin

        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end)) # Display time for forward pass 

        if i % arg.train.disp_interval == 0: # If Test interval 
            if arg.network.rot_representation=='quat':
                # input image
                inp_rgb = (rgb[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
                arg.writer.add_image('input_image', inp_rgb, i)
                cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1]) # write image 
                if 'rot' in arg.pytorch.task.lower():
                    # Save quaternion results to tensorboard
                    pred_rot_ = pred_rot[0].data.cpu().numpy().copy()
                    relative_rot_ = relative_quat[0].data.cpu().numpy().copy()
                    arg.writer.add_scalar('q_w_predicted', pred_rot_[0], i + (epoch-1) * num_iters) 
                    arg.writer.add_scalar('q_x_predicted', pred_rot_[1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_y_predicted', pred_rot_[2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_z_predicted', pred_rot_[3], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_w_target', relative_rot_[0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_x_target', relative_rot_[1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_y_target', relative_rot_[2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('q_z_target', relative_rot_[3], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('train_quat_w_err', pred_rot_[0]-relative_rot_[0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('train_quat_x_err', pred_rot_[1]-relative_rot_[1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('train_quat_y_err', pred_rot_[2]-relative_rot_[2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('train_quat_z_err', pred_rot_[3]-relative_rot_[3], i + (epoch-1) * num_iters)           
            else: 
                inp_rgb = (rgb[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
                arg.writer.add_image('input_image', inp_rgb, i)
                cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
                if 'rot' in arg.pytorch.task.lower():
                    # Save rotation matrix to tensorboard
                    pred_rot_ = pred_rot[0].data.cpu().numpy().copy()
                    relative_rot_ = relative_quat[0].data.cpu().numpy().copy()
                    arg.writer.add_scalar('r_00_predicted', pred_rot_[0,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_01_predicted', pred_rot_[0,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_02_predicted', pred_rot_[0,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_10_predicted', pred_rot_[1,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_11_predicted', pred_rot_[1,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_12_predicted', pred_rot_[1,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_20_predicted', pred_rot_[2,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_21_predicted', pred_rot_[2,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_22_predicted', pred_rot_[2,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_00_gt', relative_rot_[0,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_01_gt', relative_rot_[0,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_02_gt', relative_rot_[0,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_10_gt', relative_rot_[1,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_11_gt', relative_rot_[1,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_12_gt', relative_rot_[1,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_20_gt', relative_rot_[2,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_21_gt', relative_rot_[2,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('r_22_gt', relative_rot_[2,2], i + (epoch-1) * num_iters)
                    
                    arg.writer.add_scalar('error r_00', pred_rot_[0,0]-relative_rot_[0,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_01', pred_rot_[0,1]-relative_rot_[0,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_02', pred_rot_[0,2]-relative_rot_[0,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_10', pred_rot_[1,0]-relative_rot_[1,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_11', pred_rot_[1,1]-relative_rot_[1,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_12', pred_rot_[1,2]-relative_rot_[1,2], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_20', pred_rot_[2,0]-relative_rot_[2,0], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_21', pred_rot_[2,1]-relative_rot_[2,1], i + (epoch-1) * num_iters)
                    arg.writer.add_scalar('error r_22', pred_rot_[2,2]-relative_rot_[2,2], i + (epoch-1) * num_iters)
            if 'trans' in arg.pytorch.task.lower():
                # Save translation to tensorboard
                pred_trans_ = pred_trans[0].data.cpu().numpy().copy()
                relative_transl= relative_pose[0].data.cpu().numpy().copy()
                arg.writer.add_scalar('train_trans_x_gt', pred_trans_[0], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_y_gt', pred_trans_[1], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_z_gt', pred_trans_[2], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_x_pred', relative_transl[0], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_y_pred', relative_transl[1], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_z_pred', relative_transl[2], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_x_err', pred_trans_[0]-relative_transl[0], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_y_err', pred_trans_[1]-relative_transl[1], i + (epoch-1) * num_iters)
                arg.writer.add_scalar('train_trans_z_err', pred_trans_[2]-relative_transl[2], i + (epoch-1) * num_iters)

        # Calculate loss function 
        if 'rot' in arg.pytorch.task.lower() and not arg.network.rot_head_freeze:
            loss_rot = criterions[arg.loss.rot_loss_type](pred_rot, relative_quat_,arg.train.train_batch_size)
            loss_trans=0
        if 'trans' in arg.pytorch.task.lower() and not arg.network.trans_head_freeze:
            loss_trans = criterions[arg.loss.trans_loss_type](pred_trans, relative_pose_)
            loss_rot=0
        if 'rot_trans' in arg.pytorch.task.lower():
            loss_trans = criterions[arg.loss.trans_loss_type](pred_trans, relative_pose_)
            loss_rot = criterions[arg.loss.rot_loss_type](pred_rot, relative_quat_,arg.train.train_batch_size)

        loss = arg.loss.rot_loss_weight * loss_rot + arg.loss.trans_loss_weight * loss_trans # All loss 

        Loss.update(loss.item() if loss != 0 else 0, bs) # update loss 
        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs) # update loss_rot 
        Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs) # update loss_trans
        
        # write loss to tensorboard 
        arg.writer.add_scalar('data/loss_rot_trans', loss.item() if loss != 0 else 0, cur_iter) 
        arg.writer.add_scalar('data/loss_rot', loss_rot.item() if loss_rot != 0 else 0, cur_iter)
        arg.writer.add_scalar('data/loss_trans', loss_trans.item() if loss_trans != 0 else 0, cur_iter)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss.backward()
        optimizer.step()
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for backward of model: {}".format(T_end))
       
        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | Loss_trans {loss_trans.avg:.4f} | prediction{pred:}| real_val{totall:}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans,pred=pred_rot,totall=relative_quat)
        bar.next()
    bar.finish()
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg}, preds
