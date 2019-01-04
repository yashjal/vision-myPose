import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
sys.path.append('..')
import CocoFolder
import Mytransforms 
from utils import adjust_learning_rate as adjust_learning_rate
from utils import AverageMeter as AverageMeter
from utils import save_checkpoint as save_checkpoint
from utils import Config as Config
import pose_estimation

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--train_dir', nargs='+', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, nargs='+', type=str,
                        dest='val_dir', help='the path of val file')

    return parser.parse_args()

def construct_model(args):

    model = pose_estimation.PoseModel(num_vertices=19, num_vector=19)

    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def get_parameters(model, config, isdefault=True):

	return model.parameters(), [1.]

def my_collate(batch):
    batch = filter (lambda x:x is not None, batch)
    return torch.utils.data.dataloader.default_collate(batch)

def train_val(model, args):

    traindir = args.train_dir
    valdir = args.val_dir

    config = Config(args.config)
    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(CocoFolder.CocoFolder(traindir, 8, Mytransforms.Compose([Mytransforms.RandomResized(),\
                Mytransforms.RandomRotate(40),Mytransforms.RandomCrop(368), Mytransforms.RandomHorizontalFlip()])), batch_size=config.batch_size,\
    	shuffle=True, num_workers=config.workers, pin_memory=True)

    if config.test_interval != 0 and args.val_dir is not None:
        val_loader = torch.utils.data.DataLoader(CocoFolder.CocoFolder(valdir,8,Mytransforms.Compose([Mytransforms.TestResized(368),])),\
        	batch_size=config.batch_size, shuffle=False,num_workers=config.workers, pin_memory=True)
    
    criterion = nn.MSELoss().cuda()

    params, multiple = get_parameters(model, config, False)
    
    optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(8)]
    
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model
    learning_rate = config.base_lr

    model.train()

    heat_weight = 46 * 46 * 19 / 2.0 # for convenience to compare with origin code
    vec_weight = 46 * 46 * 38 / 2.0

    while iters < config.max_iter:
    
        for i, (input, heatmap, vecmap, mask) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            heatmap = heatmap.cuda(async=True)
            vecmap = vecmap.cuda(async=True)
            mask = mask.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            vecmap_var = torch.autograd.Variable(vecmap)
            mask_var = torch.autograd.Variable(mask)

            heat1, vec1, heat2, vec2, heat3, vec3, heat4, vec4 = model(input_var, mask_var)
            loss1_heat = criterion(heat1, heatmap_var) * heat_weight
            loss1_vec = criterion(vec1, vecmap_var) * vec_weight
            loss2_heat = criterion(heat2, heatmap_var) * heat_weight
            loss2_vec = criterion(vec2, vecmap_var) * vec_weight
            loss3_heat = criterion(heat3, heatmap_var) * heat_weight
            loss3_vec = criterion(vec3, vecmap_var) * vec_weight
            loss4_heat = criterion(heat4, heatmap_var) * heat_weight
            loss4_vec = criterion(vec4, vecmap_var) * vec_weight
 
            loss = loss1_heat + loss1_vec + loss2_heat + loss2_vec + loss3_heat + loss3_vec + loss4_heat + loss4_vec

            losses.update(loss.data[0], input.size(0))
            for cnt, l in enumerate([loss1_vec,loss1_heat,loss2_vec,loss2_heat,loss3_vec,loss3_heat,loss4_vec,loss4_heat]):
                losses_list[cnt].update(l.data[0], input.size(0))

            optimizer.zero_grad()
            #loss1_heat.backward()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(iters, config.display, learning_rate,
                    	batch_time=batch_time,data_time=data_time, loss=losses))

                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())

		#if losses.val < 90:
                	#save_checkpoint({'iter': iters,'state_dict': model.state_dict(),}, True, 'openpose_coco')
                	#break

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(8):
                    losses_list[cnt].reset()
    
            if config.test_interval != 0 and args.val_dir is not None and iters % config.test_interval == 0:

                model.eval()
                for j, (input, heatmap, vecmap, mask) in enumerate(val_loader):

                    heatmap = heatmap.cuda(async=True)
                    vecmap = vecmap.cuda(async=True)
                    mask = mask.cuda(async=True)

                    input_var = torch.autograd.Variable(input, volatile=True)
                    heatmap_var = torch.autograd.Variable(heatmap, volatile=True)
                    vecmap_var = torch.autograd.Variable(vecmap, volatile=True)
                    mask_var = torch.autograd.Variable(mask, volatile=True)

                    heat1, vec1, heat2, vec2, heat3, vec3, heat4, vec4 = model(input_var, mask_var)
                    loss1_vec = criterion(vec1, vecmap_var) * vec_weight
                    loss1_heat = criterion(heat1, heatmap_var) * heat_weight
                    loss2_heat = criterion(heat2, heatmap_var) * heat_weight
                    loss2_vec = criterion(vec2, vecmap_var) * vec_weight
                    loss3_heat = criterion(heat3, heatmap_var) * heat_weight
                    loss3_vec = criterion(vec3, vecmap_var) * vec_weight
                    loss4_heat = criterion(heat4, heatmap_var) * heat_weight
                    loss4_vec = criterion(vec4, vecmap_var) * vec_weight

                    loss = loss1_heat + loss1_vec + loss2_heat + loss2_vec + loss3_heat + loss3_vec + loss4_heat + loss4_vec

                    losses.update(loss.data[0], input.size(0))
                    for cnt, l in enumerate([loss1_vec, loss1_heat,loss2_vec, loss2_heat,loss3_vec, loss3_heat, loss4_vec, loss4_heat]):
                        losses_list[cnt].update(l.data[0], input.size(0))
    
                batch_time.update(time.time() - end)
                end = time.time()
                is_best = losses.avg < best_model
                best_model = min(best_model, losses.avg)
                save_checkpoint({'iter': iters,'state_dict': model.state_dict(),}, is_best, 'openpose_coco')
    
                print('Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.8f}\n'.format(batch_time=batch_time, loss=losses))
                
                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())
    
                batch_time.reset()
                losses.reset()

                for cnt in range(8):
                    losses_list[cnt].reset()
                
                model.train()

            # end validation
    
            if iters == config.max_iter:
            	save_checkpoint({'iter': iters,'state_dict': model.state_dict(),}, True, 'openpose_coco')
                break


if __name__ == '__main__':

    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    args = parse()
    model = construct_model(args)
    train_val(model, args)
