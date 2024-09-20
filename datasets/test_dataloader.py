# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]
            
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

# 이미지 내 다른 문장을 템플릿으로 사용
def create_templates_based_on_same_image(images):
    new_images = []
    image_groups = {}
    
    # 같은 이미지 파일을 그룹화
    for img in images:
        img_file = img[0]
        if img_file not in image_groups:
            image_groups[img_file] = []
        image_groups[img_file].append(img)
    
    # 각 그룹 내에서 템플릿을 생성
    for img_file, group in image_groups.items():
        for i, img in enumerate(group):
            # 동일한 이미지 그룹 내에서 다른 문장들을 템플릿으로 설정
            templates = [group[idx] for idx in range(len(group)) if idx != i]
            
            # 원래 데이터에 템플릿 추가
            new_img = list(img)
            new_img.append(templates[0:3])
            new_images.append(tuple(new_img))
    
    return new_images

class DatasetNotFoundError(Exception):
    pass

class GroundingDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit', 
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx=return_idx

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format('refcocogs', split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)
        # 템플릿 구성
        # self.images = create_templates_based_on_same_image(self.images)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
      if self.dataset == 'flickr':
          img_file, bbox, phrase = self.images[idx]
      else:
          img_file, _, bbox, phrase, attri, cat, templates = self.images[idx]

      ## 타겟 이미지의 bbox 처리 (target과 동일한 처리)
      if not (self.dataset == 'referit' or self.dataset == 'flickr'):
          bbox = np.array(bbox, dtype=int)
          bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
      else:
          bbox = np.array(bbox, dtype=int)

      # 타겟 이미지 처리
      img_path = osp.join(self.im_dir, img_file)
      img = Image.open(img_path).convert("RGB")
      
      # bbox를 텐서로 변환
      bbox = torch.tensor(bbox)
      bbox = bbox.float()

      # 템플릿 처리
      processed_templates = []
      for template in templates:
          temp_img_file, _, temp_bbox, temp_phrase, _ ,temp_cat = template
          
          # 템플릿 bbox 처리 (target과 동일한 방식)
          if not (self.dataset == 'referit' or self.dataset == 'flickr'):
              temp_bbox = np.array(temp_bbox, dtype=int)
              temp_bbox[2], temp_bbox[3] = temp_bbox[0] + temp_bbox[2], temp_bbox[1] + temp_bbox[3]
          else:
              temp_bbox = np.array(temp_bbox, dtype=int)

          # 템플릿 데이터 저장 (bbox 포함)
          processed_templates.append((temp_img_file, torch.tensor(temp_bbox).float(), temp_phrase, temp_cat))

      return img, phrase, bbox, processed_templates ,cat


    # def pull_item(self, idx):
    #     if self.dataset == 'flickr':
    #         img_file, bbox, phrase = self.images[idx]
    #     else:
    #         img_file, _, bbox, phrase, attri, templates = self.images[idx]
    #     ## box format: to x1y1x2y2
    #     if not (self.dataset == 'referit' or self.dataset == 'flickr'):
    #         bbox = np.array(bbox, dtype=int)
    #         bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
    #     else:
    #         bbox = np.array(bbox, dtype=int)

    #     img_path = osp.join(self.im_dir, img_file)
    #     img = Image.open(img_path).convert("RGB")
    #     # img = cv2.imread(img_path)
    #     # ## duplicate channel if gray image
    #     # if img.shape[-1] > 1:
    #     #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # else:
    #     #     img = np.stack([img] * 3)

    #     bbox = torch.tensor(bbox)
    #     bbox = bbox.float() 
    #     return img, phrase, bbox,templates

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        # return int(len(self.images) / 10)
        return len(self.images)
    def __getitem__(self, idx):
        # Target 데이터 가져오기
        img, phrase, bbox, templates , category= self.pull_item(idx)
        phrase = phrase.lower()

        # Target 데이터 전처리
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']  # 타겟 이미지 마스크
        category = category

      # BERT 텍스트 인코딩 (Target)
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

      # 템플릿 처리
        template_imgs = []
        template_img_masks = []  # 템플릿 이미지 마스크 추가
        template_word_ids = []
        template_word_masks = []
        template_bboxes = []
        template_cats = []

        for template in templates:
            temp_img_file, temp_bbox, temp_phrase, temp_cat = template[0], template[1], template[2],  template[3]
            temp_phrase = temp_phrase.lower()

            # 템플릿 이미지 변환
            temp_img_path = osp.join(self.im_dir, temp_img_file)
            temp_img = Image.open(temp_img_path).convert("RGB")
            temp_input_dict = self.transform({'img': temp_img, 'box': torch.tensor(temp_bbox).float(), 'text': temp_phrase})

            # 템플릿 이미지와 마스크 처리
            temp_img = temp_input_dict['img']
            temp_img_mask = temp_input_dict['mask']  # 템플릿 이미지 마스크 처리

            # 템플릿 텍스트 BERT 인코딩
            if self.lstm:
                temp_phrase = self.tokenize_phrase(temp_phrase)
                temp_word_id = temp_phrase
                temp_word_mask = np.array(temp_word_id > 0, dtype=int)
            else:
                examples = read_examples(temp_phrase, idx)
                features = convert_examples_to_features(examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
                temp_word_id = features[0].input_ids
                temp_word_mask = features[0].input_mask

            # 템플릿 이미지 및 텍스트 임베딩 저장
            template_imgs.append(temp_img)
            template_img_masks.append(temp_img_mask)  # 템플릿 마스크 추가
            template_word_ids.append(np.array(temp_word_id, dtype=int))
            template_word_masks.append(np.array(temp_word_mask, dtype=int))
            template_bboxes.append(np.array(temp_bbox, dtype=np.float32))
            template_cats.append(temp_cat)

        if self.testmode:
            return (img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), 
                    template_imgs, template_img_masks, template_word_ids, template_word_masks, template_bboxes, self.images[idx][0], category,template_cats)
        else:
            return (img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), 
                    template_imgs, template_img_masks, template_word_ids, template_word_masks, template_bboxes, category,template_cats)



    # def __getitem__(self, idx):
    #     img, phrase, bbox, templates = self.pull_item(idx)
    #     # phrase = phrase.decode("utf-8").encode().lower()
    #     phrase = phrase.lower()
    #     input_dict = {'img': img, 'box': bbox, 'text': phrase}
    #     input_dict = self.transform(input_dict)
    #     img = input_dict['img']
    #     bbox = input_dict['box']
    #     phrase = input_dict['text']
    #     img_mask = input_dict['mask']
        
    #     if self.lstm:
    #         phrase = self.tokenize_phrase(phrase)
    #         word_id = phrase
    #         word_mask = np.array(word_id>0, dtype=int)
    #     else:
    #         ## encode phrase to bert input
    #         examples = read_examples(phrase, idx)
    #         features = convert_examples_to_features(
    #             examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
    #         word_id = features[0].input_ids
    #         word_mask = features[0].input_mask
        
    #     if self.testmode:
    #         return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
    #             np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
    #             np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
    #     else:
    #         # print(img.shape)
    #         return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)


class GroundingDatasetCLIP(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False,):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.return_idx = return_idx

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:  ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']


        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")


        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        # return int(len(self.images) / 10)
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']

        return img, phrase, np.array(bbox, dtype=np.float32)