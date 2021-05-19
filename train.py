from __future__ import print_function
import os
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50, cfg_nas, cfg_nas320
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
import shutil
from models.retinaface import RetinaFace
from thop import profile
from eval_visualization import val_vis
from byted_nnflow.compression.torch_frame import ByteOicsrPruner

import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log_level = 'info'
log_level = getattr(logging, log_level.upper())
logging.basicConfig(level=log_level)

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./nas256_dw_aug_anchor3_hem5_wing/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
elif args.network == "nas":
    cfg = cfg_nas
elif args.network == "nas320":
    cfg = cfg_nas320

rgb_mean = (128, 128, 128) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # print('new_state_dict: {}'.format(new_state_dict))
    net.load_state_dict(new_state_dict, strict=False)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

train_path = '/opt/tiger/jupter_web/data/face_detector/train_new_visible.txt'
val_path = '/opt/tiger/jupter_web/data/face_detector/val_new_visible.txt'

val_dataset = WiderFaceDetection(training_dataset, val_path, 'val', preproc(img_dim, rgb_mean))

def val():
    net.eval()
    val_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=16, collate_fn=detection_collate)
    loss_l_all, loss_c_all, loss_landm_all, loss_vis_all = 0, 0, 0, 0
    idx = 0
    for img, target in val_loader:
        img = img.cuda()
        target = [anno.cuda() for anno in target]

        out = net(img)
        loss_l, loss_c, loss_landm, loss_vis = criterion(out, priors, target)
        loss_l_all += loss_l.item()
        loss_c_all += loss_c.item()
        loss_landm_all += loss_landm.item()
        loss_vis_all += loss_vis.item()
        idx += 1
    
    loss_l_all = loss_l_all / idx
    loss_c_all = loss_c_all / idx
    loss_landm_all = loss_landm_all / idx
    loss_vis_all = loss_vis_all / idx
    torch.cuda.empty_cache()
    return loss_l_all, loss_c_all, loss_landm_all, loss_vis_all


def train():
    
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    pattern = 'train'
    # print('rgb {}'.format(rgb_mean))
    dataset = WiderFaceDetection(training_dataset, train_path, pattern, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        net.train()
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 1 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                # if epoch > 1:
                #     shutil.rmtree('./nas_val/')
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
                val_loss_l, val_loss_c, val_loss_landm, val_loss_vis = val()
                logging.info('Validation || Epoch:{} || Loc: {:.4f} || Cla: {:.4f} || Landm: {:.4f} || Vis: {:.4f}'.format(epoch, val_loss_l, val_loss_c, val_loss_landm, val_loss_vis))
                # val_vis(net, cfg)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        # print('image {}'.format(images.shape))
        targets = [anno.cuda() for anno in targets]
        # print('landmark {}'.format(targets[2].shape))

        # forward
        out = net(images)


        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm, loss_vis = criterion(out, priors, targets)
        # print('loss vis {}'.format(loss_vis.type()))

        loss = cfg['loc_weight'] * loss_l + 1.5 * loss_c + 1.5*loss_landm + 0.5*loss_vis

        # # freeze parameters
        # loss = loss_landm

        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S +0000',
                   filename='result.log')
        logging.info('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || Visible: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), loss_vis.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')



def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
