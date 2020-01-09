import torch
import os
os.environ['TORCH_HOME'] = './'
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader 

from dorn import DORN
from dataset import AffineDataset
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import skimage.io as sio
import sys
#import CudaRender.render as render
import math
from time import time


num_epochs = 10000

parser = argparse.ArgumentParser(description='Process saome integers.')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--logtype', type=str, default='')
parser.add_argument('--save', type=str, default='')
parser.add_argument('--scale', type=int, default='2')
parser.add_argument('--eval', type=int, default='1')
parser.add_argument('--use_min', type=int, default='0')
parser.add_argument('--horizontal', type=int, default='0')
parser.add_argument('--batch_size', type=int, default='8')
parser.add_argument('--joint', type=int, default='1')
parser.add_argument('--lr', type=int, default=1e-3)
parser.add_argument('--root', type=str)
args = parser.parse_args()

batch_size = args.batch_size
train_dataset = AffineDataset(usage='train', root=args.root)
dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

test_dataset = AffineDataset(usage='test', root=args.root)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)


val_dataset = AffineDataset(usage='test', root=args.root)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

if args.save != '':
    if not os.path.exists(args.save):
        os.mkdir(args.save)
#instance of the Conv Net
cnn = DORN(channel=5,output_channel=13)
if args.logtype != '':
    writer = SummaryWriter(logdir='./dorn-resume')
#loss function and optimizer
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3);

cnn = cnn.cuda()

if (args.resume != ''):
    state = cnn.state_dict()
    state.update(torch.load(args.resume))
    cnn.load_state_dict(state)

if args.save != '':
    fp = open(args.save + '/logs.txt', 'w')
def log(str):
    if args.save != '':
        fp.write('%s\n'%(str))
        fp.flush()
    #os.fsync(fp)
    print(str)

s = 'python train_affine_dorn.py'
for j in sys.argv:
    s += ' ' + j
log(s)

def ConvertToAngle(Q):
    angle1 = torch.atan2(Q[:,1:2,:,:],Q[:,0:1,:,:]) / np.pi * 180
    angle2 = torch.atan2(Q[:,3:4,:,:],Q[:,2:3,:,:]) / np.pi * 180
    angles = torch.cat([angle1, angle2], dim=1)

    q1 = torch.cos(angle1 / 180.0 * np.pi)
    q2 = torch.sin(angle1 / 180.0 * np.pi)

    return angles

def ConvertToDirection(Q):
    x1 = torch.cos(Q[:,0:1,:,:] / 180.0 * np.pi)
    y1 = torch.sin(Q[:,0:1,:,:] / 180.0 * np.pi)
    x2 = torch.cos(Q[:,1:2,:,:] / 180.0 * np.pi)
    y2 = torch.sin(Q[:,1:2,:,:] / 180.0 * np.pi)

    return torch.cat([x1,y1,x2,y2], dim=1)

def Rotate90(Q):
    Q0 = Q.clone()
    Q0[:,0,:,:] = Q[:,2,:,:]
    Q0[:,1,:,:] = Q[:,3,:,:]
    Q0[:,2,:,:] = -Q[:,0,:,:]
    Q0[:,3,:,:] = -Q[:,1,:,:]
    return Q0

def RemoveAngleAmbiguity(Q0):
    Q = Q0.clone()
    mask = Q[:,0:1,:,:] > 90
    mask = torch.cat([mask, mask], dim=1)
    while torch.sum(mask).item() > 0:
        Q -= 90 * mask.float()
        mask = Q[:,0:1,:,:] > 90
        mask = torch.cat([mask, mask], dim=1)
    mask = Q[:,0:1,:,:] < 0
    mask = torch.cat([mask, mask], dim=1)
    while torch.sum(mask).item() > 0:
        Q += 90 * mask.float()
        mask = Q[:,0:1,:,:] < 0
        mask = torch.cat([mask, mask], dim=1)

    mask = (Q > 45).float()
    return mask * (90 - Q) + (1 - mask) * Q, Q

n_iter = 0
m_iter = 0

test_iter = iter(test_dataloader)
val_iter = iter(val_dataloader)

def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-7).view(dir_x.shape[0],1,dir_x.shape[2],dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l

def train_one_iter(i, sample_batched, evaluate=0):
    global n_iter, m_iter
    cnn.train()
    if evaluate > 0 and args.eval == 1:
        cnn.eval()
    images = sample_batched['image']
    labels = sample_batched['label']
    labels_alt = sample_batched['label_alt']
    mask_alt = sample_batched['mask']
    tmask = mask_alt.clone()
    X = sample_batched['X']
    Y = sample_batched['Y']
    
    images_tensor = Variable(images.float())
    labels_tensor = Variable(labels)
    labels_alt_tensor = Variable(labels_alt)
    mask_alt_tensor = Variable(mask_alt)

    images_tensor, labels_tensor, labels_alt_tensor, mask_alt_tensor = images_tensor.cuda(), labels_tensor.cuda(), labels_alt_tensor.cuda(), mask_alt_tensor.cuda()
    
    mask = (labels_tensor[:,0:1,:,:] * labels_tensor[:,0:1,:,:] + labels_tensor[:,1:2,:,:] * labels_tensor[:,1:2,:,:]) > 0.2
    if args.horizontal == 0:
        mask_alt_tensor = ((mask_alt_tensor < 0.9) & (mask_alt_tensor > 0.1)).view(mask_alt_tensor.shape[0],1,mask_alt_tensor.shape[1],mask_alt_tensor.shape[2])
        mask = mask & mask_alt_tensor
    elif args.horizontal == 1:
        mask_alt_tensor = ((mask_alt_tensor > 0.9)).view(mask_alt_tensor.shape[0],1,mask_alt_tensor.shape[1],mask_alt_tensor.shape[2])
        mask = mask & mask_alt_tensor

    mask = mask.float()
    X = sample_batched['X'].cuda()
    Y = sample_batched['Y'].cuda()

    elems = torch.sum(mask).item() * 2
    if elems == 0:
        return
    # Forward + Backward + Optimize
    optimizer.zero_grad()
    
    outputs_temp = cnn(images_tensor)
    outputs = outputs_temp[:,0:4,:,:]
    outputs2 = outputs_temp[:,4:10,:,:]
    norm1 = outputs_temp[:,10:13,:,:]
    dir_x = Normalize(outputs2[:,0:3,:,:])
    dir_y = Normalize(outputs2[:,3:6,:,:])

    preds = ConvertToAngle(outputs)

    l0 = labels_tensor
    a1 = ConvertToAngle(l0)
    l1 = Rotate90(l0)
    a2 = ConvertToAngle(l1)
    l2 = Rotate90(l1)
    a3 = ConvertToAngle(l2)
    l3 = Rotate90(l2)
    a4 = ConvertToAngle(l3)

    if args.use_min == 0:
        d0 = preds - a1
        d0 = torch.min(torch.abs(d0), torch.min(torch.abs(d0 + 360), torch.abs(d0 - 360)))
        d0 = torch.sum(d0, dim=1)
        d0 = d0.view(d0.shape[0], 1, d0.shape[1], d0.shape[2])
        d = d0 * mask
        loss = torch.sum(d)
        diff = (outputs - l0) ** 2
        diff = torch.sum(diff, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff = diff * mask
        mse_loss = torch.sum(diff)

        diff_2a = torch.sum((dir_x - X) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2b = torch.sum((dir_y - Y) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        mse_loss_2 = torch.sum((diff_2a + diff_2b) * mask)

    else:
        d0 = preds - a1
        d0 = torch.min(torch.abs(d0), torch.min(torch.abs(d0 + 360), torch.abs(d0 - 360)))
        d0 = torch.sum(d0, dim=1)
        d1 = preds - a2
        d1 = torch.min(torch.abs(d1), torch.min(torch.abs(d1 + 360), torch.abs(d1 - 360)))
        d1 = torch.sum(d1, dim=1)
        d2 = preds - a3
        d2 = torch.min(torch.abs(d2), torch.min(torch.abs(d2 + 360), torch.abs(d2 - 360)))
        d2 = torch.sum(d2, dim=1)
        d3 = preds - a4
        d3 = torch.min(torch.abs(d3), torch.min(torch.abs(d3 + 360), torch.abs(d3 - 360)))
        d3 = torch.sum(d3, dim=1)
        d = torch.min(d0, torch.min(d1, torch.min(d2, d3)))
        d = d.view(d.shape[0], 1, d.shape[1], d.shape[2])
        d = d * mask
        loss = torch.sum(d)

        diff1 = torch.sum((outputs - l0) ** 2, dim=1)
        diff2 = torch.sum((outputs - l1) ** 2, dim=1)
        diff3 = torch.sum((outputs - l2) ** 2, dim=1)
        diff4 = torch.sum((outputs - l3) ** 2, dim=1)
        diff = torch.min(diff1, torch.min(diff2, torch.min(diff3, diff4)))
        diff = diff.view(diff.shape[0], 1, diff.shape[1], diff.shape[2])        
        mse_loss = torch.sum(diff * mask)

        diff_2a = torch.sum((dir_x - X) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2b = torch.sum((dir_y - Y) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2_x = diff_2a + diff_2b

        diff_2a = torch.sum((dir_x - Y) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2b = torch.sum((dir_y + X) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2_y = diff_2a + diff_2b

        diff_2a = torch.sum((dir_x + X) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2b = torch.sum((dir_y + Y) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2_z = diff_2a + diff_2b

        diff_2a = torch.sum((dir_x + Y) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2b = torch.sum((dir_y - X) ** 2, dim=1).view(outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])
        diff_2_w = diff_2a + diff_2b

        diff_2 = torch.min(diff_2_x, torch.min(diff_2_y, torch.min(diff_2_z, diff_2_w)))
        mse_loss_2 = torch.sum(diff_2 * mask)

    c_1 = dir_x[:,0,:,:] - images_tensor[:,3,:,:] * dir_x[:,2,:,:] - outputs[:,0,:,:]
    c_2 = dir_x[:,1,:,:] - images_tensor[:,4,:,:] * dir_x[:,2,:,:] - outputs[:,1,:,:]
    c_3 = dir_y[:,0,:,:] - images_tensor[:,3,:,:] * dir_y[:,2,:,:] - outputs[:,2,:,:]
    c_4 = dir_y[:,1,:,:] - images_tensor[:,4,:,:] * dir_y[:,2,:,:] - outputs[:,3,:,:]
    mse_loss_proj = torch.sum((c_1 ** 2 + c_2 ** 2 + c_3 ** 2 + c_4 ** 2).view(mask.shape[0],1,mask.shape[2],mask.shape[3]) * mask)

    norm0 = Normalize(torch.cross(dir_x, dir_y, dim=1))
    norm1 = Normalize(norm1)
    norm2 = Normalize(torch.cross(X,Y,dim=1))

    angle = torch.acos(torch.clamp(torch.sum(norm1 * norm2, dim=1), -1, 1)) / np.pi * 180
    angle = angle.view(mask.shape[0],1,mask.shape[2],mask.shape[3]) * mask
    angle = torch.sum(angle)

    mse_loss_norm = torch.sum(torch.sum((norm1 - norm0)**2,dim=1).view(mask.shape[0],1,mask.shape[2],mask.shape[3]) * mask)
    angle_loss = torch.sum(torch.sum((norm1 - norm2)**2, dim=1).view(mask.shape[0],1,mask.shape[2],mask.shape[3]) * mask)
    if args.train == 0:
        preds = ConvertToDirection(preds)
        #labels_tensor = ConvertToDirection(labels_tensor)
        mask = torch.cat([mask, mask, mask, mask], dim=1).float()
        #preds *= mask
        labels_tensor *= mask
        for j in range(preds.shape[0]):
            im = images_tensor[j].data.cpu().numpy()
            im = np.ascontiguousarray(np.swapaxes(np.swapaxes(im, 0, 1), 1, 2))
            pred = preds[j].data.cpu().numpy()
            pred = np.ascontiguousarray(np.swapaxes(np.swapaxes(pred, 0, 1), 1, 2))
            label = labels_tensor[j].data.cpu().numpy()
            label = np.ascontiguousarray(np.swapaxes(np.swapaxes(label, 0, 1), 1, 2))
            m = tmask[j].numpy()
            m = (m * 255).astype('uint8')
            label = label / np.max(np.abs(label))
            pred = pred / np.max(np.abs(pred))
            sio.imsave('preds/pred-%06d-color.png'%(m_iter*preds.shape[0]+j), im[:,:,0:3])


            normal = norm2[j].data.cpu().numpy()
            normal = np.ascontiguousarray(np.swapaxes(np.swapaxes(normal, 0, 1), 1, 2))
            sio.imsave('preds/pred-%06d-normal-gt.png'%(m_iter*preds.shape[0]+j), normal * 0.5 + 0.5)
            normal_pred = norm1[j].data.cpu().numpy()
            normal_pred = np.ascontiguousarray(np.swapaxes(np.swapaxes(normal_pred, 0, 1), 1, 2))
            sio.imsave('preds/pred-%06d-normal-pred.png'%(m_iter*preds.shape[0]+j), normal_pred * 0.5 + 0.5)

            diff = normal_pred - normal
            diff = (np.sqrt(np.sum(diff * diff, axis=2)) * 512).astype('uint8') * (m > 0)
            sio.imsave('preds/pred-%06d-normal-diff.png'%(m_iter*preds.shape[0]+j), diff)
            #sio.imsave('preds/pred-%06d-mask.png'%(m_iter*preds.shape[0]+j), (mask[j][0].data.cpu().numpy() * 255).astype('uint8'))

            color = np.ascontiguousarray(im[:,:,0:3]).astype('float32')
            #try:
            Qx = np.ascontiguousarray(label[:,:,0:2].astype('float32'))
            Qy = np.ascontiguousarray(label[:,:,2:4].astype('float32'))
            #render.visualizeDirection('preds/pred-%06d-vis-gt.png'%(m_iter*preds.shape[0]+j), color, Qx, Qy)


            diff1 = torch.sum((outputs - l0) ** 2, dim=1)
            diff2 = torch.sum((outputs - l1) ** 2, dim=1)
            diff3 = torch.sum((outputs - l2) ** 2, dim=1)
            diff4 = torch.sum((outputs - l3) ** 2, dim=1)
            diff = torch.min(diff1, torch.min(diff2, torch.min(diff3, diff4)))
            mask1 = (diff == diff1).data.cpu().numpy()[j]
            mask2 = (diff == diff2).data.cpu().numpy()[j]
            mask3 = (diff == diff3).data.cpu().numpy()[j]
            mask4 = (diff == diff4).data.cpu().numpy()[j]

            mask1 = np.tile(np.reshape(mask1, (mask1.shape[0],mask1.shape[1],1)), (1,1,2))
            mask2 = np.tile(np.reshape(mask2, (mask2.shape[0],mask2.shape[1],1)), (1,1,2))
            mask3 = np.tile(np.reshape(mask3, (mask3.shape[0],mask3.shape[1],1)), (1,1,2))
            mask4 = np.tile(np.reshape(mask4, (mask4.shape[0],mask4.shape[1],1)), (1,1,2))

            Qx1 = np.ascontiguousarray(pred[:,:,0:2].astype('float32'))
            Qy1 = np.ascontiguousarray(pred[:,:,2:4].astype('float32'))

            Qx2 = np.ascontiguousarray(pred[:,:,2:4].astype('float32'))
            Qy2 = np.ascontiguousarray(-pred[:,:,0:2].astype('float32'))

            Qx3 = np.ascontiguousarray(-pred[:,:,0:2].astype('float32'))
            Qy3 = np.ascontiguousarray(-pred[:,:,2:4].astype('float32'))

            Qx4 = np.ascontiguousarray(-pred[:,:,2:4].astype('float32'))
            Qy4 = np.ascontiguousarray(pred[:,:,0:2].astype('float32'))

            Qx = Qx1 * mask1 + Qx2 * mask2 + Qx3 * mask3 + Qx4 * mask4
            Qy = Qy1 * mask1 + Qy2 * mask2 + Qy3 * mask3 + Qy4 * mask4

        m_iter += 1
    else:
        if evaluate == 0:
            if args.joint == 1:
                losses = mse_loss + mse_loss_2 + angle_loss + mse_loss_proj * 5 + mse_loss_norm * 5# + angle / 200.0
            elif args.joint == 0:
                losses = angle_loss
            elif args.joint == 2:
                losses = angle_loss + mse_loss_2
            elif args.joint == 3:
                losses = angle_loss + mse_loss_2 + mse_loss
            elif args.joint == 4:
                losses = angle_loss + mse_loss_2 + mse_loss + mse_loss_norm * 5
            elif args.joint == 5:
                losses = angle_loss + mse_loss_2 + mse_loss + mse_loss_proj * 5
            losses.backward()
            optimizer.step()

        if args.logtype != '':
            if evaluate == 0:
                writer.add_scalar(args.logtype + '/project_loss', mse_loss.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/3D_loss', mse_loss_2.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/Consistency_loss', mse_loss_proj.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/Normal_Consistency_loss', mse_loss_norm.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/projection_err', loss.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/normal_err', angle.item() / elems * 2, n_iter)
                n_iter += 1
            elif evaluate == 1:
                writer.add_scalar(args.logtype + '/val_loss', mse_loss.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/val_loss2', mse_loss_2.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/val_loss3', mse_loss_proj.item() / elems, n_iter)
                writer.add_scalar(args.logtype + '/val_err', loss.item() / elems, m_iter)
                m_iter += 1
            else:
                writer.add_scalar(args.logtype + '/test_project_loss', mse_loss.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/test_3D_loss', mse_loss_2.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/test_Consistency_loss', mse_loss_proj.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/test_Normal_Consistency_loss', mse_loss_norm.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/test_projection_err', loss.item() / elems, m_iter)
                writer.add_scalar(args.logtype + '/test_normal_err', angle.item() / elems * 2, m_iter)
                m_iter += 1

    if evaluate == 0:
        log ('Epoch : %d/%d, Iter : %d/%d,  Loss: <%.4f, %.4f, %.4f>,  Err: <%.4f %.4f>' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, mse_loss.item()/elems, mse_loss_2.item()/elems, mse_loss_proj.item()/elems, loss.item() / elems, angle.item() / elems * 2))
    else:
        log ('(Test) Epoch : %d/%d, Iter : %d/%d,  Loss: <%.4f, %.4f, %.4f>,  Err: <%.4f %.4f>' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, mse_loss.item()/elems, mse_loss_2.item()/elems, mse_loss_proj.item()/elems, loss.item() / elems, angle.item() / elems * 2))
    del images_tensor, labels_tensor, loss, outputs, mask

for epoch in range(num_epochs):
    if args.train == 1:
        for i, sample_batched in enumerate(dataloader):
            #print('start train')
            if i % 8 == 0:
                m_iter += 1
                try:
                    sample_batched_t = next(test_iter)
                except:
                    test_iter = iter(test_dataloader)
                    sample_batched_t = next(test_iter)
                train_one_iter(i, sample_batched_t, 2)

            train_one_iter(i, sample_batched, 0)

            if i % 1000 == 0 and args.save != '':
                path = args.save + '/model-epoch-%05d-iter-%05d.cpkt'%(epoch, i)
                torch.save(cnn.state_dict(), path)

    if args.train == 0:
        correct_num = 0
        total_num = 0
        for i, sample_batched in enumerate(test_dataloader):
            train_one_iter(i, sample_batched, 2)
            m_iter += 1
        break

    if epoch == 2:
        args.use_min = 1
        args.horizontal = 2     

writer.close()
