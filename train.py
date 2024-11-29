import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from dotmap import DotMap

from network import MFDyNet
# from network import DANet as MFDyNet
from utils.logger import MYLog
from utils.eyeq_simple_dataset import EyeQTrainDataset, EyeQUnpairTrainDataset
from utils.eyeq_simple_dataset import EyeQSyntheticTestDataset
from utils.utils import format_time
from utils.metrics import eval_metrics as my_metrics
from utils.myloss import GANLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_loss

import argparse


class EnhancementFW():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        if cfgs.framework == 'MFDyNet':
            self.generator = MFDyNet.MFDyNet(32, 4).to('cuda')
        elif cfgs.framework == 'MFNet':
            self.generator = MFDyNet.MFNet(32, 4).to('cuda')
        elif cfgs.framework == 'DyNet':
            self.generator = MFDyNet.DyNet(32, 4).to('cuda')
        elif cfgs.framework == 'Baseline3':
            self.generator = MFDyNet.Baseline3Net(32, 4).to('cuda')
        elif cfgs.framework == 'Baseline1':
            self.generator = MFDyNet.Baseline1Net(32, 4).to('cuda')
        
        print(self.generator)
        time.sleep(2)
            
        
        self.discriminator = MFDyNet.Discriminator(in_channels=3, use_sigmoid=cfgs.gan_loss != 'hinge').to('cuda')
        self.re_loss = nn.L1Loss().to('cuda')
        self.gan_loss = GANLoss(cfgs.gan_loss).to('cuda')
        self.ssim_loss = ssim_loss(data_range=1.0).to('cuda')
        #self.randomcrop = torchvision.transforms.RandomCrop(180)
        
        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(cfgs.init_lr),
            betas=(cfgs.beta1, cfgs.beta2),
        )
        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(cfgs.init_lr) * float(cfgs.d2g_lr),
            betas=(cfgs.beta1, cfgs.beta2),
        )
                                                                                                           
        self.gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.gen_optimizer, T_max=cfgs.T_max, eta_min=cfgs.eta_min)
        self.dis_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.dis_optimizer, T_max=cfgs.T_max, eta_min=cfgs.eta_min)
        

    def load(self,path):
        network_checkpoint = torch.load(path)
        self.generator.load_state_dict(network_checkpoint['gen'])
        self.discriminator.load_state_dict(network_checkpoint['dis'])

    def train(self):
        self.generator.train()
        self.discriminator.train()
        
    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
    
    def batch_pass(self, batch):
        # lq_images, hq_images, de_array, r_hq_images

        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        
        # process outputs
        gen_out = self.forward((batch["lq"], batch['lq_hsv'], batch['lq_lab']))
        # gen_out = self.forward(batch["lq"])
        gen_loss = 0
        dis_loss = 0 
        
        # discriminator loss
        # dis_input_real = torch.cat([lq_images, hq_images], dim=1)
        # dis_input_fake = torch.cat([lq_images, gen_out.detach()], dim=1)
        dis_input_real = batch['hq']
        dis_input_fake = gen_out.detach()
        # dis_input_fake = batch["lq"]
        
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.gan_loss(dis_real, True, True)
        dis_fake_loss = self.gan_loss(dis_fake, False, True)
        dis_rf_loss =  (dis_real_loss + dis_fake_loss) / 2
        dis_loss += dis_rf_loss 
         
        # generator adversarial loss
        gen_input_fake = gen_out
        gen_fake, _ = self.discriminator(gen_input_fake)                                                  # in: [rgb(3)]
        gen_gan_loss = self.gan_loss(gen_fake, True, False) * self.cfgs.gan_loss_w
        gen_loss += gen_gan_loss
        
        # generator l1 loss
        gen_re_loss = self.re_loss(gen_out, batch['hq']) * self.cfgs.re_loss_w
        gen_loss += gen_re_loss
        gen_ssim_loss = 1-self.ssim_loss(gen_out, batch['hq'])
        gen_loss += gen_ssim_loss
        
        logs = [
            ("l_dis", dis_rf_loss.item()),
            ("l_gan", gen_gan_loss.item()),
            ("l_re", gen_re_loss.item()),
            ('l_ssim', gen_ssim_loss.item()),
        ]
        
                                                                                                           
        return gen_out, gen_loss, dis_loss, logs

    
    def forward(self, inputs):
        outputs = self.generator(*inputs)
        # outputs = self.generator(inputs)
        return outputs
    
    def backward(self, gen_loss=None, dis_loss=None):
        gen_loss.backward()
        self.gen_optimizer.step()
        
        dis_loss.backward()
        self.dis_optimizer.step()
        
        self.gen_scheduler.step()
        self.dis_scheduler.step()
        
class myModel():
    def __init__(self, cfgs, comments, len_traindl, train_loader, test_loader, mode='train'):
        self.iteration = 0
        self.total_time = 0
        self.total_iteration = 0
        self.cfgs = cfgs
        
        self.model = EnhancementFW(cfgs)
        
        if mode == 'train':
            print('Training mode')
            self.len_traindl = len_traindl
            self.mylog = MYLog(dataset='EyeQ', cfgs=cfgs, \
                                         comments=comments, val=True)

            self.train_metrics = my_metrics(disable_fun=True)
            self.eval_metrics = my_metrics()
            self.lq_metrics = my_metrics()
            
        if mode == 'eval':
            print('Evaluation mode')
            self.eval_metrics = my_metrics()
            self.lq_metrics = my_metrics()

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.SINCE_TIME = 0 
        
    def train(self):
        #
        print('Training start...')
        print()
        self.model.train()
        dataloader = self.train_loader
        #training per epoch
        
        for epoch in range(1, self.cfgs.epochs+1):
            #training per batch
            
            self.iteration = 0
            self.SINCE_TIME = time.time()
            
            for batch in dataloader:
                #forward
                for key in batch:
                    if key != 'name':
                        batch[key] = batch[key].cuda()
                gen_out, gen_loss, dis_loss, logs = self.model.batch_pass(batch)
                
                #backward
                self.model.backward(gen_loss=gen_loss, dis_loss=dis_loss)
                
                self.iteration += 1
                self.total_iteration += 1
                END_TIME = time.time()
                self.total_time += (END_TIME - self.SINCE_TIME)
                
                #display output, and save output per 200 iterations
                if self.iteration % 200 == 0:
                    #title
                    self.mylog.log(f'Running Time: {round(END_TIME-self.SINCE_TIME)}s, total: {format_time(self.total_time)}', stage='train')
                    self.mylog.log(f'Epoch[{epoch}|{self.cfgs.epochs}][{self.iteration}/{self.len_traindl}]', stage='train')

                    #loss funciton
                    loss_s = []
                    for lossname, lossvalue in logs:
                        loss_s.append(f'{lossname}: {lossvalue:.4f}') 

                    loss_s = '  '.join(loss_s)
                    self.mylog.log(loss_s, stage='train')     
                    
                    #evaluation metrics
                    eval_s = f'PSNR: {self.train_metrics.batchpsnr(gen_out, batch["hq"]):.4f}'
                    eval_s = eval_s + '   '
                    eval_s = eval_s + f'SSIM: {self.train_metrics.batchssim(gen_out, batch["hq"]):.4f}'                    
                    self.mylog.log(eval_s, stage='train')
                    self.mylog.log('', stage='train')
                    
                    
                    for i in range(len(gen_out)):
                        img_name  = f'{epoch}_{self.iteration}_{i}_lq_hq_fakehq.png'
                        lq_hq_fakehq = torch.cat([batch["lq"][i], batch["hq"][i], gen_out[i]], dim=2)  
                        self.mylog.log_output(lq_hq_fakehq, img_name, stage='val')
                
                else:
                    print(f'Running Time: {round(END_TIME-self.SINCE_TIME)}s, total: {format_time(self.total_time)}')
                    print(f'Epoch[{epoch}|{self.cfgs.epochs}][{self.iteration}/{self.len_traindl}]')
                    loss_s = []
                    for lossname, lossvalue in logs:
                        loss_s.append(f'{lossname}: {lossvalue:.4f}') 

                    loss_s = '  '.join(loss_s)
                    print(loss_s)   
                    
                #self.model.scheduler.step()
                self.SINCE_TIME = time.time()
                                                                                                           
            #save network checkpoint per cp_interval
            if epoch % self.cfgs.cp_interval == 0 or epoch == self.cfgs.epochs:
                self.mylog.save_net({'gen': self.model.generator.state_dict(), 
                                    'dis': self.model.discriminator.state_dict(),},
                                     'MFDynet_'+str(epoch)+'_net.pt')         
                print(f'Net save at epoch {epoch}')
            
            if epoch % self.cfgs.eval_interval == 0 or epoch == self.cfgs.epochs:
                self.eval(self.test_loader, epoch)
                self.model.train()
                
            
            
        print()
        print('Training finish...')
    
    
    def load(self, net_index, net_epoch):
        cp_path = os.path.join('./myoutput', \
                               str(net_index).zfill(5), \
                               'EyeQ', \
                               'checkpoint', \
                               'MFDynet_' + str(net_epoch)+'_net.pt')
        self.model.load(cp_path)
        
        print(f'Loading net on {net_index} and {net_epoch}.')
        print()
    
    def eval(self, dataloader, epoch):
        print('Evaluation...')
        print()
        self.model.eval()
        
        #final
        self.eval_metrics.empty()
        self.lq_metrics.empty()
        
        
        with torch.no_grad():
            for batch in dataloader:
                for key in batch:
                    if key != 'name':
                        batch[key] = batch[key].cuda()
                fake_hq_imgs = self.model.forward((batch["lq"], batch['lq_hsv'], batch['lq_lab']))
                
                #calculate generated quality
                self.eval_metrics.op(fake_hq_imgs, batch["hq"])
                
                #calculate degraded quality
                self.lq_metrics.op(batch["lq"], batch["hq"])
                #break
        #final
        self.eval_metrics.final()
        self.lq_metrics.final()
        
        self.mylog.log(f'Evaluation at epoch{epoch}:', stage='val')
        self.mylog.log(f'Degraded:', stage='val')
        self.mylog.log(f'PSNR: {self.lq_metrics.psnr:.4f}.  SSIM: {self.lq_metrics.ssim:.4f}', stage='val')
        self.mylog.log(f'Enhanced:', stage='val')
        self.mylog.log(f'PSNR: {self.eval_metrics.psnr:.4f}.  SSIM: {self.eval_metrics.ssim:.4f}', stage='val')
        self.mylog.log('', stage='val')
        

if __name__ == '__main__':
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)

    parser = argparse.ArgumentParser(description='Backbone and framework selection.')
    parser.add_argument('--framework', type=str, required=True,
                    help='Selection framework: [MFDyNet, MFNet, DyNet, Net]')
    parser.add_argument('--epochs', type=int, default=150,
                    help='training epoch')
    parser.add_argument('--batchsize', type=int, required=True,
                    help='batchsize, [4]')    
    parser.add_argument('--resize', type=int, required=True,
                    help='') 
    
    
    parser.add_argument('--gan_loss', type=str, default='nsgan', required=True,
                    help='loss type')        

    parser.add_argument('--gan_loss_w', type=float, default=0.1, required=True,
                    help='loss weight')        
    parser.add_argument('--d2g_lr', type=float, default=0.1, required=True,
                    help='ratio')        
                                                                                                           
                                                                                                           
    parser.add_argument('--load_index', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    parser.add_argument('--load_epoch', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    
                                                                                                           
    
    args = parser.parse_args()
#baseline < dfm < clf
    cfgs = {'resize'    : args.resize, 
            'if_aug'    : True, 
            'if_norm'   : True,
            'init_lr'   : 0.0001,                   # learning rate
            'dataset_path'      : './dataset/EyeQ',
            'beta1'             : 0.9,                # 0 or 0.9
            'beta2'             : 0.9,
            'cp_interval'       : 2, 
            'eval_interval'     : 10,
            'seed'              : 2023,
            'eta_min'           : 1e-7,
            'gan_loss'          : args.gan_loss,
            
            #training detial 
            
            #loss weight
            're_loss_w'         : 1,
            'gan_loss_w'        : args.gan_loss_w,
            'd2g_lr'            : args.d2g_lr,
            
            #framework
            'epochs'            : args.epochs,
            'load_index'        : args.load_index,
            'load_epoch'        : args.load_epoch,
            'bs'                : args.batchsize,
            'framework'         : args.framework,       
            }

    comments = ''
    cfgs = DotMap(cfgs, _dynamic=False)

    trainingloader = DataLoader(EyeQTrainDataset(cfgs), batch_size=cfgs.bs, shuffle=True)
    testloader = DataLoader(EyeQSyntheticTestDataset(cfgs), batch_size=1, shuffle=False)
    
    cfgs.T_max = len(trainingloader) * cfgs.epochs

    DANetformer = myModel(cfgs, comments, len(trainingloader), trainingloader, testloader)
    if cfgs.load_epoch != 0:
        DANetformer.load(cfgs.load_index, cfgs.load_epoch)
    DANetformer.train()
