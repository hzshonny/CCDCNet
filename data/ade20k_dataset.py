# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        # if is_train:
        #     parser.set_defaults(load_size=286)
        # else:
        #     parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=1) # 150 # 255 # 2
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' else 'train'
        subfolder = 'test' if opt.phase == 'test' else 'train'
        cache = False if opt.phase == 'test' else True
        all_images = sorted(make_dataset(root + '/' + subfolder, recursive=True, read_cache=cache, write_cache=False))
        image_paths = []
        label_paths = []
        for p in all_images:
            if '_%s' % phase not in p:  # _%s_
                continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = '_test111' if opt.phase == 'test' else '' # _test1
        with open('./data/knife_ref{}.txt'.format(extra)) as fd:  # scissors ade20k knife gdxray
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('train', 'test')
        return ref_dict, train_test_folder

