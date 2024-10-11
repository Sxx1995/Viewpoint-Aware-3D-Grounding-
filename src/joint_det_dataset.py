# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

import csv
from collections import defaultdict
import h5py
import json
import multiprocessing as mp
import os
import random
from six.moves import cPickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
import wandb

from data.model_util_scannet import ScannetDatasetConfig
from data.scannet_utils import read_label_mapping
from src.visual_data_handlers import Scan
from .scannet_classes import REL_ALIASES, VIEW_DEP_RELS
from tqdm import tqdm
from copy import deepcopy

NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132


class Joint3DDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset_dict={'sr3d': 1, 'scannet': 10},
                 test_dataset='sr3d',
                 split='train', overfit=False,
                 data_path='./',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 butd=False, butd_gt=False, butd_cls=False, augment_det=False,
                 num_angles = 36):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.num_angles = num_angles
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == 'train'
        self.use_multiview = use_multiview
        self.data_path = data_path
        self.visualize = False  # manually set this to True to debug
        self.butd = butd
        self.butd_gt = butd_gt
        self.butd_cls = butd_cls
        self.joint_det = (  # joint usage of detection/grounding phrases
            'scannet' in dataset_dict
            and len(dataset_dict.keys()) > 1
            and self.split == 'train'
        )
        self.augment_det = augment_det

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        self.label_map = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )
        self.label_map18 = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40id'
        )
        self.label_mapclass = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40class'
        )
        self.multiview_path = os.path.join(
            f'{self.data_path}/scanrefer_2d_feats',
            "enet_feats_maxpool.hdf5"
        )
        self.multiview_data = {}
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        if os.path.exists('data/cls_results.json'):
            with open('data/cls_results.json') as fid:
                self.cls_results = json.load(fid)  # {scan_id: [class0, ...]}

        # load
        print('Loading %s files, take a breath!' % split)
        print(f'{self.data_path}/{split}_v3scans.pkl')
        if not os.path.exists(f'{self.data_path}/{split}_v3scans.pkl'):
            print(zzzzzzzzzzzzzz)
            save_data(f'{data_path}/{split}_v3scans.pkl', split, data_path)
        
        print(test_dataset, split, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.scans = unpickle_data(f'{self.data_path}/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]
        self.annos_list = {}
        if self.split != 'train':
            self.annos = self.load_annos(test_dataset)
        else:
            for dset, cnt in dataset_dict.items():
                _annos = self.load_annos(dset)
                self.annos_list[dset] = [_annos, cnt]
            self.build_dataset()
        if self.visualize:
            wandb.init(project="vis", name="debug")

    def build_dataset(self):
        self.annos = []
        for dset in self.annos_list:
            print(dset, len(self.annos_list[dset][0]), self.annos_list[dset][1])
            if self.annos_list[dset][1] >= 1:
                self.annos += (self.annos_list[dset][0] * int(self.annos_list[dset][1]))
            elif self.annos_list[dset][1] > 0 and self.annos_list[dset][1] < 1:
                count = int(self.annos_list[dset][1] * len(self.annos_list[dset][0]))
                idx = np.random.choice(np.arange(0, len(self.annos_list[dset][0])), size = count, replace = False)
                self.annos += [self.annos_list[dset][0][i] for i in idx] 

    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            'nr3d': self.load_nr3d_annos,
            'sr3d': self.load_sr3d_annos,
            'sr3d+': self.load_sr3dplus_annos,
            'scanrefer': self.load_scanrefer_annos,
            'syn_scanrefer': self.load_scanrefer_annos,
            'scannet': self.load_scannet_annos,
            'syn_data': self.load_scanrefer_annos
        }
        if dset == 'syn_scanrefer':
            print(dset, dset == 'syn_scanrefer')
            annos = loaders[dset]('syn_')
        else:
            print(dset, dset == 'scanrefer')
            annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos

    def load_sr3dplus_annos(self):
        """Load annotations of sr3d/sr3d+."""
        return self.load_sr3d_annos(dset='sr3d+')

    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self.data_path + 'sr3d_pred_spans.json', 'r') as f:
            pred_spans = json.load(f)
        with open( 'DATA_ROOT/scanrefer/global_appeared_objects.json') as f:
            self.used_obj = json.load(f)
        #self.used_obj = [x.replace('_00', '') for x in self.used_obj]
        print(self.data_path + 'refer_it_3d/%s_%s.csv' % (dset, split))
        csv_reader = csv.reader(open(self.data_path + 'refer_it_3d/%s_%s.csv' % (dset, split)))
        #csv_reader = csv.reader(open(self.data_path + 'refer_it_3d/%s.csv' % (dset)))
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}
        annos = []
        a = 0
        b = 0
        with tqdm(total=50000) as pbar:
            for i, anno in enumerate(csv_reader):
                pbar.update(1)
                if anno[headers['scan_id']] in scan_ids and \
                  str(anno[headers['mentions_target_class']]).lower() == 'true':
                    if not '%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']])) in self.used_obj:
                        self.used_obj.append('%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']])))
                    """
                    if not '%s-%d'%(anno[headers['scan_id']][:-3], int(anno[headers['target_id']])) in self.used_obj:
                        self.used_obj.append('%s-%d'%(anno[headers['scan_id']][:-3], int(anno[headers['target_id']])))
                        #print('%s-%d'%(anno[headers['scan_id']][:-3], int(anno[headers['target_id']])))
                        a += 1
                    else:
                        b += 1
                    """
                    #print(a, b, a/(a+b))
                    for j in range(len(pred_spans)):
                        if pred_spans[j]['utterance'] == anno[headers['utterance']]:
                             annos += [
                                 {
                                     'scan_id': anno[headers['scan_id']],
                                     'target_id': int(anno[headers['target_id']]),
                                     #'G_target_id': self.used_obj.index('%s-%d'%(anno[headers['scan_id']][:-3], int(anno[headers['target_id']]))) if split == 'train' else -1,
                                     'G_target_id': self.used_obj.index('%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']]))) if split == 'train' else -1,
                                     'distractor_ids': [],
                                     'utterance': anno[headers['utterance']],
                                     'target': anno[headers['instance_type']],
                                     'anchors': eval(anno[headers['anchors_types']]),
                                     'anchor_ids': eval(anno[headers['anchor_ids']]),
                                     'dataset': 'sr3d',
                                     'pred_pos_map': pred_spans[j]['span'],  # predicted span
                                     'span_utterance': pred_spans[j]['utterance'],  # for assert
                                     'num_angles': -1,
                                     'sampled_angle': [0] * 36,
                                     'sampled_center':  [[[0] * 10]*10],
                                     #'rotated_y_axis': anno["rotated_y_axis"],
                                     'appear_objects': {"name": [], "indx": []}
                                 }]
                             break
        for i in annos:
            assert 'num_angles' in i.keys()
            assert 'G_target_id' in i.keys()
        return annos

    def load_nr3d_annos(self):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(self.data_path + 'nr3d_pred_spans.json', 'r') as f:
            pred_spans = json.load(f)
        with open( 'DATA_ROOT/scanrefer/global_appeared_objects.json') as f:
            self.used_obj = json.load(f)
        csv_reader = csv.reader(open(self.data_path + 'refer_it_3d/nr3d.csv'))
        headers = next(csv_reader)
        headers = {header: h for h, header in enumerate(headers)}
        annos = []
        for i, anno in enumerate(csv_reader):
            if anno[headers['scan_id']] in scan_ids and \
              str(anno[headers['mentions_target_class']]).lower() == 'true' and \
              (
                 str(anno[headers['correct_guess']]).lower() == 'true' or \
                 split != 'test' \
              ):
                 if not '%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']])) in self.used_obj:
                     self.used_obj.append('%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']])))
                 annos += [
                    {
                        'scan_id': anno[headers['scan_id']],
                        'target_id': int(anno[headers['target_id']]),
                        'G_target_id': self.used_obj.index('%s-%d'%(anno[headers['scan_id']], int(anno[headers['target_id']]))) if split == 'train' else -1,
                        'distractor_ids': [],
                        'utterance': anno[headers['utterance']],
                        'target': anno[headers['instance_type']],
                        'distractor_ids': -1,
                        'anchors': [],
                        'anchor_ids': [],
                        'dataset': 'nr3d',
                        'pred_pos_map': pred_spans[i]['span'],  # predicted span
                        'span_utterance': pred_spans[i]['utterance'],  # for assert
                        'num_angles': -1,
                        'sampled_angle': [0] * 36,
                        'sampled_center':  [[[0] * 10]*10],
                        #'rotated_y_axis': anno["rotated_y_axis"],
                        'appear_objects': {"name": [], "indx": []}
                    }]
        for i in annos:
            assert 'num_angles' in i.keys()
            assert 'G_target_id' in i.keys()

        # Add distractor info
        for anno in annos:
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if self.scans[anno['scan_id']].get_object_instance_label(ind)
                == anno['target']
                and ind != anno['target_id']
            ]
        return annos

    def load_scanrefer_annos(self, data_type = ''):
        """Load annotations of ScanRefer."""
        path_base = 'scanrefer/%sScanRefer_filtered'%(data_type)
        _path = self.data_path + path_base
        split = self.split
        if split in ('val', 'test'):
            split = 'val'
        with open(self.data_path + f'%sscanrefer_pred_spans_{split}.json'%(data_type)) as f:
            pred_spans = json.load(f)
        with open(_path + '_%s.txt' % split) as f:
            scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
        with open(_path + '_%s.json' % split) as f:
            reader = json.load(f)
        with open( 'DATA_ROOT/scanrefer/global_appeared_objects.json') as f:
            self.used_obj = json.load(f)
        annos = []
        for i, anno in enumerate(reader[:len(reader)]):
            if anno['scene_id'] in scan_ids:
                 if "num_angles" not in anno.keys():
                     anno["num_angles"] = -1
                     #anno["sampled_angle"]= 0
                     anno["sampled_angle"]= [0] * 36
                     #anno["sampled_center"] = [0, 0]
                     anno["sampled_center"] = [[0] * 10]*10
                     #anno["rotated_y_axis"] = [[0, 0]]
                     anno["appear_objects"] = {"name": [], "indx": []}

                 annos += [
                    {
                        'object_name': anno['object_name'],
                        'scan_id': anno['scene_id'],
                        'target_id': int(anno['object_id']),
                        'G_target_id': self.used_obj.index('%s-%d'%(anno['scene_id'], int(anno['object_id']))) if split == 'train' else -1,
                        'distractor_ids': [],
                        'utterance': ' '.join(anno['token']),
                        'target': ' '.join(str(anno['object_name']).split('_')),
                        'anchors': [],
                        'anchor_ids': [],
                        'dataset': 'scanrefer',
                        'pred_pos_map': pred_spans[i]['span'],  # predicted span
                        'span_utterance': pred_spans[i]['utterance'],  # for assert
                        'num_angles': anno["num_angles"],
                        'sampled_angle': anno["sampled_angle"],
                        'sampled_center': [anno["sampled_center"]],
                        #'rotated_y_axis': anno["rotated_y_axis"],
                        'appear_objects': anno["appear_objects"]
                    }]
        for i in annos:
            assert 'num_angles' in i.keys()
            assert 'G_target_id' in i.keys()
            #assert len(i['rotated_y_axis']) == 1 and len(i['rotated_y_axis'][0]) == 2, i['rotated_y_axis']
            #assert len(i['sampled_center']) == 1 and len(i['sampled_center'][0]) == 2, i['sampled_center']
        # Add distractor info
        scene2obj = defaultdict(list)
        sceneobj2used = defaultdict(list)
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if labels[ind] == labels[anno['target_id']]
                and ind != anno['target_id']
            ][:32]
            if anno['target_id'] not in sceneobj2used[anno['scan_id']]:
                sceneobj2used[anno['scan_id']].append(anno['target_id'])
                scene2obj[anno['scan_id']].append(labels[anno['target_id']])
        # Add unique-multi
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['unique'] = (
                np.array(scene2obj[anno['scan_id']])
                == labels[anno['target_id']]
            ).sum() == 1
        return annos

    def load_scannet_annos(self):
        """Load annotations of scannet."""
        split = 'train' if self.split == 'train' else 'val'
        with open('data/meta_data/scannetv2_%s.txt' % split) as f:
            scan_ids = [line.rstrip() for line in f]
        annos = []
        for scan_id in scan_ids:
            scan = self.scans[scan_id]
            # Ignore scans that have no object in our vocabulary
            keep = np.array([
                self.label_map[
                    scan.get_object_instance_label(ind)
                ] in DC.nyu40id2class
                for ind in range(len(scan.three_d_objects))
            ])
            if keep.any():
                # this will get populated randomly each time
                annos.append({
                    'scan_id': scan_id,
                    'target_id': [],
                    'distractor_ids': [],
                    'utterance': '',
                    'target': [],
                    'anchors': [],
                    'anchor_ids': [],
                    'dataset': 'scannet',
                    'num_angles': -1,
                    'G_target_id': -1,
                    #'sampled_angle': 0,
                    'sampled_angle': [0] * 36,
                    #'sampled_center': [[0, 0]],
                    'sampled_center': [[[0] * 10] * 10],
                    #'rotated_y_axis': [[0, 0]],
                    'appear_objects': {"name": [], "indx": []},
                })
        if self.split == 'train':
            annos = [
                anno for a, anno in enumerate(annos)
                if a not in {965, 977}
            ]
        return annos


    def _sample_classes(self, scan_id):
        """Sample classes for the scannet detection sentences."""
        scan = self.scans[scan_id]
        sampled_classes = set([
            self.label_map[scan.get_object_instance_label(ind)]
            for ind in range(len(scan.three_d_objects))
        ])
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))
        # sample 10 classes
        if self.split == 'train' and self.random_utt:  # random utterance
            if len(sampled_classes) > 10:
                sampled_classes = random.sample(sampled_classes, 10)
            ret = [DC.class2type[DC.nyu40id2class[i]] for i in sampled_classes]
            random.shuffle(ret)
        else:  # fixed detection sentence
            ret = [
                'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'other furniture'
            ]
        return ret

    def _create_scannet_utterance(self, sampled_classes):
        if self.split == 'train' and self.random_utt:
            neg_names = []
            while len(neg_names) < 10:
                _ind = np.random.randint(0, len(DC.class2type))
                if DC.class2type[_ind] not in neg_names + sampled_classes:
                    neg_names.append(DC.class2type[_ind])
            mixed_names = sorted(list(set(sampled_classes + neg_names)))
            random.shuffle(mixed_names)
        else:
            mixed_names = sampled_classes
        utterance = ' . '.join(mixed_names)
        return utterance

    def _load_multiview(self, scan_id):
        """Load multi-view data of given scan-id."""
        pid = mp.current_process().pid
        if pid not in self.multiview_data:
            self.multiview_data[pid] = h5py.File(
                self.multiview_path, "r", libver="latest"
            )
        return self.multiview_data[pid][scan_id]

    def _augment(self, pc, color, rotate):
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90 * np.random.randint(0, 4) + 10 * np.random.rand() - 5
            # Flipping along the YZ plane
            augmentations['yz_flip'] = np.random.random() > 0.5
            if augmentations['yz_flip']:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations['xz_flip'] = np.random.random() > 0.5
            if augmentations['xz_flip']:
                pc[:, 1] = -pc[:, 1]
        else:
            theta_z = (2 * np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2 * np.random.rand() - 1) * 2.5
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2 * np.random.rand() - 1) * 2.5
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        augmentations['scale'] = 0.98 + 0.04 * np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04 * np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations

    def _get_pc(self, anno, scan):
        """Return a point cloud representation of current scene."""
        scan_id = anno['scan_id']
        rel_name = "none"
        if anno['dataset'].startswith('sr3d'):
            rel_name = self._find_rel(anno['utterance'])

        # a. Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # b .Height
        height = None
        if self.use_height:
            floor_height = np.percentile(scan.pc[:, 2], 0.99)
            height = np.expand_dims(scan.pc[:, 2] - floor_height, 1)

        # c. Multi-view 2d features
        multiview_data = None
        if self.use_multiview:
            multiview_data = self._load_multiview(scan_id)

        # d. Augmentations
        augmentations = {}
        if self.split == 'train' and self.augment:
            rotate_natural = (
                anno['dataset'] in ('nr3d', 'scanrefer')
                and self._augment_nr3d(anno['utterance'])
            )
            rotate_sr3d = (
                anno['dataset'].startswith('sr3d')
                and rel_name not in VIEW_DEP_RELS
            )
            rotate_else = anno['dataset'] == 'scannet'
            rotate = rotate_sr3d or rotate_natural or rotate_else
            pc, color, augmentations = self._augment(scan.pc, color, rotate)
            scan.pc = pc

        # e. Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)
        if height is not None:
            point_cloud = np.concatenate([point_cloud, height], 1)
        if multiview_data is not None:
            point_cloud = np.concatenate([point_cloud, multiview_data], 1)

        return point_cloud, augmentations, color

    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        if self.detect_intermediate:
            cat_names += anno['anchors']
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span + 1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        return tokens_positive, positive_map

    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        if isinstance(anno['target_id'], list):  # scannet
            tids = anno['target_id']
        else:  # referit dataset
            tids = [anno['target_id']]
            if self.detect_intermediate:
                tids += anno.get('anchor_ids', [])
        point_instance_label = -np.ones(len(scan.pc))
        for t, tid in enumerate(tids):
            point_instance_label[scan.three_d_objects[tid]['points']] = t

        bboxes[:len(tids)] = np.stack([
            scan.get_object_bbox(tid).reshape(-1) for tid in tids
        ])
        bboxes = np.concatenate((
            (bboxes[:, :3] + bboxes[:, 3:]) * 0.5,
            bboxes[:, 3:] - bboxes[:, :3]
        ), 1)
        if self.split == 'train' and self.augment:  # jitter boxes
            bboxes[:len(tids)] *= 0.95 + 0.1 * np.random.random((len(tids), 6))
        bboxes[len(tids):, :3] = 1000
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1
        return bboxes, box_label_mask, point_instance_label

    def _get_scene_objects(self, scan):
        # Objects to keep
        keep_ = np.array([
            self.label_map[
                scan.get_object_instance_label(ind)
            ] in DC.nyu40id2class
            for ind in range(len(scan.three_d_objects))
        ])[:MAX_NUM_OBJ]
        keep = np.array([False] * MAX_NUM_OBJ)
        keep[:len(keep_)] = True  # keep_

        # Class ids
        cid = np.array([
            DC.nyu40id2class[self.label_map[scan.get_object_instance_label(k)]]
            if keep_[k] else 325  # this is the 'object' class
            for k, kept in enumerate(keep) if kept
        ])
        class_ids = np.zeros((MAX_NUM_OBJ,))
        class_ids[keep] = cid

        # Object boxes
        all_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_bboxes_ = np.stack([
            scan.get_object_bbox(k).reshape(-1)
            for k, kept in enumerate(keep) if kept
        ])
        # cx, cy, cz, w, h, d
        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)
        all_bboxes[keep] = all_bboxes_
        if self.split == 'train' and self.augment:
            all_bboxes *= 0.95 + 0.1 * np.random.random((len(all_bboxes), 6))

        # Which boxes we're interested for
        all_bbox_label_mask = keep
        return class_ids, all_bboxes, all_bbox_label_mask

    def _get_detected_objects(self, split, scan_id, augmentations):
        # Initialize
        all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
        detected_class_ids = np.zeros((MAX_NUM_OBJ,))
        detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

        # Load
        detected_dict = np.load(
            f'{self.data_path}group_free_pred_bboxes_{split}/{scan_id}.npy',
            allow_pickle=True
        ).item()

        all_bboxes_ = np.array(detected_dict['box'])
        classes = detected_dict['class']
        cid = np.array([DC.nyu40id2class[
            self.label_map[c]] for c in detected_dict['class']
        ])

        all_bboxes_ = np.concatenate((
            (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
            all_bboxes_[:, 3:] - all_bboxes_[:, :3]
        ), 1)

        assert len(classes) < MAX_NUM_OBJ
        assert len(classes) == all_bboxes_.shape[0]

        num_objs = len(classes)
        all_detected_bboxes[:num_objs] = all_bboxes_
        all_detected_bbox_label_mask[:num_objs] = np.array([True] * num_objs)
        detected_class_ids[:num_objs] = cid
        detected_logits[:num_objs] = detected_dict['logits']
        # Match current augmentations
        if self.augment and self.split == 'train':
            all_det_pts = box2points(all_detected_bboxes).reshape(-1, 3)
            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])
            if augmentations.get('yz_flip', False):
                all_det_pts[:, 0] = -all_det_pts[:, 0]
            if augmentations.get('xz_flip', False):
                all_det_pts[:, 1] = -all_det_pts[:, 1]
            all_det_pts += augmentations['shift']
            all_det_pts *= augmentations['scale']
            all_detected_bboxes = points2box(all_det_pts.reshape(-1, 8, 3))
        if self.augment_det and self.split == 'train':
            min_ = all_detected_bboxes.min(0)
            max_ = all_detected_bboxes.max(0)
            rand_box = (
                (max_ - min_)[None]
                * np.random.random(all_detected_bboxes.shape)
                + min_
            )
            corrupt = np.random.random(len(all_detected_bboxes)) > 0.7
            all_detected_bboxes[corrupt] = rand_box[corrupt]
            detected_class_ids[corrupt] = np.random.randint(
                0, len(DC.nyu40ids), (len(detected_class_ids))
            )[corrupt]
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        )

    def __getitem__(self, index):
        """Get current batch for input index."""
        split = self.split

        s = []
        # Read annotation
        anno = self.annos[index]
        scan = self.scans[anno['scan_id']]
        scan.pc = np.copy(scan.orig_pc)
        # Populate anno (used only for scannet)
        e = deepcopy(anno)
        self.random_utt = False
        if anno['dataset'] == 'scannet':
            self.random_utt = self.joint_det and np.random.random() > 0.5
            sampled_classes = self._sample_classes(anno['scan_id'])
            utterance = self._create_scannet_utterance(sampled_classes)
            # Target ids
            accured_pc = np.ones(50000)
            if not self.random_utt:  # detection18 phrase
                anno['target_id'] = np.where(np.array([
                    self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] in DC18.nyu40id2class
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            else:
                anno['target_id'] = np.where(np.array([
                    self.label_map[
                        scan.get_object_instance_label(ind)
                    ] in DC.nyu40id2class
                    and
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]] in sampled_classes
                    for ind in range(len(scan.three_d_objects))
                ])[:MAX_NUM_OBJ])[0].tolist()
            anno['G_target_id'] = -1 #self.used_obj.index('%s-%d'%(anno['scan_id'], int(anno['target_id'][0])))  
            # Target names
            if not self.random_utt:
                anno['target'] = [
                    DC18.class2type[DC18.nyu40id2class[self.label_map18[
                        scan.get_object_instance_label(ind)
                    ]]]
                    if self.label_map18[
                        scan.get_object_instance_label(ind)
                    ] != 39
                    else 'other furniture'
                    for ind in anno['target_id']
                ]
            else:
                anno['target'] = [
                    DC.class2type[DC.nyu40id2class[self.label_map[
                        scan.get_object_instance_label(ind)
                    ]]]
                    for ind in anno['target_id']
                ]
            anno['utterance'] = utterance

        else:
            accured_pc = np.zeros(50000)
            for ii, i in enumerate(anno['appear_objects']['indx']):
                #print(scan.three_d_objects[i]['points'])
                #print(zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz)
                s += scan.three_d_objects[i]['points'].tolist()
                accured_pc[scan.three_d_objects[i]['points']] = 1
                #print(scan.get_object_instance_label(i), anno['appear_objects']['name'][ii])
                assert scan.get_object_instance_label(i) == anno['appear_objects']['name'][ii], print(i, anno)
        if accured_pc.sum() == 0: 
            accured_pc = np.ones(50000)
        assert accured_pc.sum() > 0, [anno, s]
        # Point cloud representation
        point_cloud, augmentations, og_color = self._get_pc(anno, scan)

        # "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan)

        # Positive map for soft-token and contrastive losses
        if anno['dataset'] == 'scannet':
            _, positive_map = self._get_token_positive_map(anno)
        else:
            assert anno['utterance'] == anno['span_utterance']  # sanity check
            positive_map = np.zeros((MAX_NUM_OBJ, 256))
            positive_map_ = np.array(anno['pred_pos_map']).reshape(-1, 256)
            positive_map[:len(positive_map_)] = positive_map_

        # Scene gt boxes
        (
            class_ids, all_bboxes, all_bbox_label_mask
        ) = self._get_scene_objects(scan)

        # Detected boxes
        (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        ) = self._get_detected_objects(split, anno['scan_id'], augmentations)

        # Assume a perfect object detector
        if self.butd_gt:
            all_detected_bboxes = all_bboxes
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = class_ids

        # Assume a perfect object proposal stage
        if self.butd_cls:
            all_detected_bboxes = all_bboxes
            all_detected_bbox_label_mask = all_bbox_label_mask
            detected_class_ids = np.zeros((len(all_bboxes,)))
            classes = np.array(self.cls_results[anno['scan_id']])
            # detected_class_ids[all_bbox_label_mask] = classes[classes > -1]
            classes[classes == -1] = 325  # 'object' class
            _k = all_bbox_label_mask.sum()
            detected_class_ids[:_k] = classes[:_k]

        # Visualize for debugging
        if self.visualize and anno['dataset'].startswith('sr3d'):
            self._visualize_scene(anno, point_cloud, og_color, all_bboxes)

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        if not isinstance(anno['target_id'], int) and not self.random_utt:
            _labels[:len(anno['target_id'])] = np.array([
                DC18.nyu40id2class[self.label_map18[
                    scan.get_object_instance_label(ind)
                ]]
                for ind in anno['target_id']
            ])
        ret_dict = {
            'obj_name_id': self.label_map18[scan.get_object_instance_label(anno['target_id'])] if not anno['dataset'] == 'scannet' else -1,
            'box_label_mask': box_label_mask.astype(np.float32),
            'center_label': gt_bboxes[:, :3].astype(np.float32),
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32),
            'num_angles': np.array(anno["num_angles"]).astype(np.float32),
            'sampled_angle': np.array(anno["sampled_angle"]).astype(np.float32),
            'sampled_center': np.array(anno["sampled_center"]).astype(np.float32),
            #'rotated_y_axis': np.array(anno["rotated_y_axis"]).astype(np.float32),
            'acured_pc_label': accured_pc,
            #'num_angles': anno['num_angles'], 
            #'sampled_angle': anno['sampled_angle']
        }
        #assert ret_dict['num_angles'] == self.num_angles
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "G_target_id": anno['G_target_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (
                ' '.join(anno['utterance'].replace(',', ' ,').split())
                + ' . not mentioned'
            ),
            "positive_map": positive_map.astype(np.float32),
            "relation": (
                self._find_rel(anno['utterance'])
                if anno['dataset'].startswith('sr3d')
                else "none"
            ),
            "target_name": scan.get_object_instance_label(
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "target_id": (
                anno['target_id'] if isinstance(anno['target_id'], int)
                else anno['target_id'][0]
            ),
            "point_instance_label": point_instance_label.astype(np.int64),
            "all_bboxes": all_bboxes.astype(np.float32),
            "all_bbox_label_mask": all_bbox_label_mask.astype(np.bool8),
            "all_class_ids": class_ids.astype(np.int64),
            "distractor_ids": np.array(
                anno['distractor_ids']
                + [-1] * (32 - len(anno['distractor_ids']))
            ).astype(int),
            "anchor_ids": np.array(
                anno['anchor_ids']
                + [-1] * (32 - len(anno['anchor_ids']))
            ).astype(int),
            "all_detected_boxes": all_detected_bboxes.astype(np.float32),
            "all_detected_bbox_label_mask": all_detected_bbox_label_mask.astype(np.bool8),
            "all_detected_class_ids": detected_class_ids.astype(np.int64),
            "all_detected_logits": detected_logits.astype(np.float32),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            "target_cid": (
                class_ids[anno['target_id']]
                if isinstance(anno['target_id'], int)
                else class_ids[anno['target_id'][0]]
            )
        })
        return ret_dict

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        words = set(utterance.split())
        return any(rel in words for rel in rels)

    @staticmethod
    def _find_rel(utterance):
        utterance = ' ' + utterance.replace(',', ' ,') + ' '
        relation = "none"
        sorted_rel_list = sorted(REL_ALIASES, key=len, reverse=True)
        for rel in sorted_rel_list:
            if ' ' + rel + ' ' in utterance:
                relation = REL_ALIASES[rel]
                break
        return relation

    @staticmethod
    def _augment_nr3d(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across', 'top', 'below', 'under'
        ]
        augment = True
        for rel in rels:
            if ' ' + rel + ' ' in (utterance + ' '):
                augment = False
        return augment

    def _visualize_scene(self, anno, point_cloud, og_color, all_bboxes):
        target_id = anno['target_id']
        distractor_ids = np.array(
            anno['distractor_ids']
            + [-1] * (10 - len(anno['distractor_ids']))
        ).astype(int)
        anchor_ids = np.array(
            anno['anchor_ids']
            + [-1] * (10 - len(anno['anchor_ids']))
        ).astype(int)
        point_cloud[:, 3:] = (og_color + self.mean_rgb) * 256
        predict_center = np.array([[-3.4511173, -4.644377 ,  0.]])
        rotate_matrix = np.array([[ 0.86602557,  0.49999976,  0.        ],
                                  [-0.49999976,  0.86602557,  0.        ],
                                  [ 0.        ,  0.        ,  1.        ]])
        point_cloud[:, :3] = np.matmul(point_cloud[:, :3] - predict_center, rotate_matrix)

        centers = np.random.rand(1000,3)
        centers_dis = ((centers**2).sum(-1)**0.5)[:, np.newaxis]
        centers = centers / centers_dis 
        center_color = np.ones(centers.shape) * 256
        centers = np.concatenate([centers, center_color], axis = 1)
        point_cloud = np.concatenate([point_cloud, centers], axis = 0)

        all_boxes_points = box2points(all_bboxes[..., :6])

        target_box = all_boxes_points[target_id]
        anchors_boxes = all_boxes_points[[
            i.item() for i in anchor_ids if i != -1
        ]]
        distractors_boxes = all_boxes_points[[
            i.item() for i in distractor_ids if i != -1
        ]]

        wandb.log({
            "ground_truth_point_scene": wandb.Object3D({
                "type": "lidar/beta",
                "points": point_cloud,
                "boxes": np.array(
                    [  # target
                        {
                            "corners": target_box.tolist(),
                            "label": "target",
                            "color": [0, 255, 0]
                        }
                    ]
                    + [  # anchors
                        {
                            "corners": c.tolist(),
                            "label": "anchor",
                            "color": [0, 0, 255]
                        }
                        for c in anchors_boxes
                    ]
                    + [  # distractors
                        {
                            "corners": c.tolist(),
                            "label": "distractor",
                            "color": [0, 255, 255]
                        }
                        for c in distractors_boxes
                    ]
                    + [  # other
                        {
                            "corners": c.tolist(),
                            "label": "other",
                            "color": [255, 0, 0]
                        }
                        for i, c in enumerate(all_boxes_points)
                        if i not in (
                            [target_id]
                            + anchor_ids.tolist()
                            + distractor_ids.tolist()
                        )
                    ]
                )
            }),
            "utterance": wandb.Html(anno['utterance']),
            "scan_id": wandb.Html(anno['scan_id']),
        })

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)


def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T


def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)


def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    return Scan(scan_id, scan_path, True)


def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    scan_path = data_path + 'scans/'
    with open('data/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]
    print('{} scans found.'.format(len(scan_ids)))

    # Load data
    n_items = len(scan_ids)
    n_processes = 4  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path)
        for scan_id in scan_ids
    ]
    DATA = []
    for i in tqdm(iter_obj):
    	DATA.append(scannet_loader(i))
    	print(DATA[-1])
    i = 0
    for data in tqdm(DATA):
        all_scans[scan_ids[i]] = data
        i += 1
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, all_scans)


def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()
