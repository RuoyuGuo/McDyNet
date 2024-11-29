import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from dotmap import DotMap
from PIL import Image
from tqdm import tqdm

from network import MFDyNet
from utils.logger import MYLog
from utils.utils import format_time, tensor2np
from utils.metrics import eval_metrics as my_metrics
from utils.myloss import GANLoss
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

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

    def load(self,path):
        network_checkpoint = torch.load(path)
        self.generator.load_state_dict(network_checkpoint['gen'])

    def train(self):
        self.generator.train()
        
    def eval(self):
        self.generator.eval()

    def forward(self, inputs):
        outputs = self.generator(*inputs)
        # outputs = self.generator(inputs)
        return outputs
        
class myModel():
    def __init__(self, cfgs):
        self.iteration = 0
        self.total_time = 0
        self.total_iteration = 0
        self.cfgs = cfgs
        
        self.model = EnhancementFW(cfgs)      
        self.SINCE_TIME = 0 
        
    
    def load(self, net_index, net_epoch):
        cp_path = os.path.join('./myoutput', \
                               str(net_index).zfill(5), \
                               'EyeQ', \
                               'checkpoint', \
                               'MFDynet_' + str(net_epoch)+'_net.pt')
        self.model.load(cp_path)
        
        print(f'Loading net on {net_index} and {net_epoch}.')
        print()
    
    def eval(self, dataloader):
        print('Evaluation...')
        print()
        self.model.eval()
        
        eh_psnr = 0
        eh_ssim = 0 
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                for key in batch:
                    if key != 'name':
                        batch[key] = batch[key].cuda()
                
                fake_hq_imgs = self.model.forward((batch["lq"], batch['lq_hsv'], batch['lq_lab']))

                
                img = fake_hq_imgs.squeeze(0).permute(1,2,0).cpu().numpy() * 255
                img = np.clip(img, 0, 255).astype(np.uint8)
                #img.save(os.path.join(cfgs.output_path, filename[0]))
                
                if self.cfgs.cal_metric:
                    eh = fake_hq_imgs.squeeze(0).permute(1,2,0).cpu().numpy()
                    gt = batch["hq"].squeeze(0).permute(1,2,0).cpu().numpy()
                    
                    eh = np.clip(eh, 0, 1)
                    gt = np.clip(gt, 0, 1)
                    
                    temp_psnr = peak_signal_noise_ratio(eh, gt, data_range=1.0)
                    temp_ssim = structural_similarity(eh, gt, channel_axis=2, data_range=1.0)

                    eh_psnr += temp_psnr
                    eh_ssim += temp_ssim
                    count += 1
                
                if self.cfgs.save_output:
                    cv.imwrite(os.path.join(cfgs.output_path, str(batch['name'][0])), img[:,:,::-1])
                
        print(f'total psnr {eh_psnr/count:.4f}, total ssim {eh_ssim/count:.4f}. {count}')
        
if __name__ == '__main__':
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)

    parser = argparse.ArgumentParser(description='Backbone and framework selection.')
    parser.add_argument('--framework', type=str, required=True,
                    help='Selection framework: [baseline, dfm, clf]]')
    

    parser.add_argument('--load_index', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    parser.add_argument('--load_epoch', type=int, required=True,
                    help='load pretrain model, 0 for training from scratch')    
    
    #path
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True,
                    help='dataset path')    
    parser.add_argument('--output_path', type=str, required=True,
                    help='save path')   
    
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--cal_metric' , action='store_true')
    
      
    args = parser.parse_args()

    cfgs = {'resize'    : 256, 
            'if_aug'    : False, 
            'if_norm'   : True,
            'dataset_path'      : args.dataset_path, 
            'dataset'           : args.dataset, 
            'output_path'       : args.output_path,
            
            #framework
            'load_index'        : args.load_index,
            'load_epoch'        : args.load_epoch,
            'framework'         : args.framework,       #baseline < dfm < clf
            
            'save_output'       : args.save_output,
            'cal_metric'        : args.cal_metric,
            
            }

    comments = ''
    cfgs = DotMap(cfgs, _dynamic=False)
        
    if cfgs.dataset == 'eyeqtest':
        from utils.eyeq_simple_dataset import EyeQTestDataset as TestDataset
    if cfgs.dataset == 'eyequsable':
        from utils.eyeq_simple_dataset import EyeQUsableDataset as TestDataset
    if cfgs.dataset == 'eyeqreject':
        from utils.eyeq_simple_dataset import EyeQRejectDataset as TestDataset
    if cfgs.dataset == 'drive':
        from utils.drive_dataset import MFDRIVETestDataset as TestDataset
    if cfgs.dataset == 'refuge':
        from utils.refuge_dataset import REFUGETestDataset as TestDataset
            
    testloader = DataLoader(TestDataset(cfgs), batch_size=1, shuffle=False)
    
    DANetformer = myModel(cfgs)
    DANetformer.load(cfgs.load_index, cfgs.load_epoch)
    DANetformer.eval(testloader)
