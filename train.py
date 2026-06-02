# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
from torchvision.utils import save_image
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.util import print_current_errors
from util.util import mkdir
from trainers.pix2pix_trainer import Pix2PixTrainer
"""
 python train.py  --PONO --PONO_C --vgg_normal_correct  --video_like  --nThreads 16 --amp --display_winsize 256 --load_size 286  --crop_size 256  --label_nc 3 --batchSize 4  --gpu_ids 1 --netG dynast --niter 100 --niter_decay 100 --vgg_path vgg/vgg19_conv.pth --n_layers 3 --use_atten --continue_train --contrastive_weight 1.0 --style_weight 0.  --continue_train

 python train.py  --PONO --PONO_C --vgg_normal_correct  --video_like  --nThreads 16 --amp --display_winsize 256 --load_size 286  --crop_size 256  --label_nc 3 --batchSize 4  --gpu_ids 2 --netG dynast --niter 100 --niter_decay 100 --vgg_path vgg/vgg19_conv.pth --n_layers 3 --use_atten --contrastive_weight 10.0 --style_weight 1.0 

"""
if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))
    dataloader = data.create_dataloader(opt)
    len_dataloader = len(dataloader)
    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))
    # create trainer for our model
    trainer = Pix2PixTrainer(opt, resume_epoch=iter_counter.first_epoch)
    # save_root = os.path.join(opt.checkpoints_dir, opt.name, 'train')
    save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.name)
    mkdir(save_root)

    for epoch in iter_counter.training_epochs():
        opt.epoch = epoch
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)
            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                try:
                    print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                         iter_counter.epoch_iter_num, losses, iter_counter.time_per_iter)
                except OSError as err:
                    print(err)

            if iter_counter.needs_displaying():
                imgs_num = data_i['label'].shape[0]
                if opt.dataset_mode == 'celebahq':
                    data_i['label'] = data_i['label'][:, ::2, :, :]
                elif opt.dataset_mode == 'celebahqedge':
                    data_i['label'] = data_i['label'][:, :1, :, :]
                elif opt.dataset_mode == 'deepfashion':
                    data_i['label'] = data_i['label'][:, :3, :, :]
                if data_i['label'].shape[1] == 3:
                    label = data_i['label']
                else:
                    label = data_i['label'].expand(-1, 3, -1, -1).float() / data_i['label'].max()

                show_size = opt.display_winsize
                # imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), trainer.out['warp_out'].cpu(),
                #                       trainer.get_latest_generated().data.cpu(), data_i['image'].cpu()), 0)
                imgs = torch.cat((label.cpu(), data_i['ref'].cpu(),
                                  trainer.get_latest_generated().data.cpu(),
                                  data_i['image'].cpu()), 0)

                try:
                    save_name = '%08d_%08d.jpg' % (epoch, iter_counter.total_steps_so_far)
                    save_name = os.path.join(save_root, save_name)
                    save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
                except OSError as err:
                    print(err)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                try:
                    trainer.save('latest')
                    iter_counter.record_current_iter()
                except OSError as err:
                    import pdb

                    pdb.set_trace()
                    print(err)

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                trainer.save(epoch)
            except OSError as err:
                print(err)

    print('Training was successfully finished.')

'''
python train.py --name ade20k --dataset_mode ade20k --dataroot ./data/knife_rotation --niter 100  
--niter_decay 100 --use_attention --maskmix --warp_mask_losstype direct --weight_mask 100.0 --PONO --PONO_C --batchSize 2 
--vgg_normal_correct --gpu_ids 2 --vgg_path /models/vgg19_conv.pth --netG dynast


python train.py --name ade20k --dataset_mode ade20k --PONO --PONO_C --amp --batchSize 4 --netG dynast --load_size 
256 --crop_size 256 --dataroot /data/knife_rotation --contrastive_weight 100.0 --label_nc 1 --niter 
 100 --niter_decay 100 --gpu_ids 0 --use_atten --vgg_normal_correct --style_weight 0.1  --weight_warp_self 1000.0 
--weight_perceptual 0.001 --vgg_path ./models/vgg19_conv.pth --continue_train '''

'''python train.py --name ade20k --dataset_mode ade20k --dataroot ./dataset/knife
--niter 100 --niter_decay 100  --PONO --PONO_C --batchSize 4 --vgg_normal_correct --vgg_path models/vgg19_conv.pth
 --gpu_ids 0
'''

'''--mcl --nce_w=0.4'''

##########
# --contrastive_weight 100.0


########### --PONO --PONO_C
'''python train.py --name ade20k --dataset_mode ade20k --dataroot /dataset/knife
--niter 100 --niter_decay 100  --batchSize 4 --vgg_normal_correct --vgg_path models/vgg19_conv.pth
 --gpu_ids 0
 --mcl --nce_w=0.4 --contrastive_weight 100.0
'''


