import os
import re
import torch
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random

# InputExample 클래스: BERT에 입력할 예시를 생성하는 클래스
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

# InputFeatures 클래스: 변환된 BERT 입력의 특징을 저장하는 클래스
class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

# 문장을 파싱해서 InputExample로 변환하는 함수
def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line.strip()
    text_a, text_b = None, None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples

# BERT 입력으로 변환하는 함수
def convert_examples_to_features(examples, seq_length, tokenizer):
    """Converts a set of `InputExample`s into `InputFeatures`."""
    features = []
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # tokens_a와 tokens_b의 길이를 seq_length에 맞게 조정
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[:seq_length - 2]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_type_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # 패딩 적용
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
    return features

# 두 문장 길이 제한 함수
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# 데이터셋 클래스
class SupportGroundingDataset(Dataset):
    """
    SupportGroundingDataset: 텍스트와 이미지를 로드하고 BERT에 맞게 전처리된 데이터를 제공하는 PyTorch Dataset 클래스.
    클래스별로 필터링하지 않고, 주어진 데이터에서 랜덤하게 샘플링하는 방식으로 동작합니다.

    Args:
        data_root (str): 이미지 파일이 저장된 최상위 경로.
        split_root (str): 데이터셋의 학습/검증/테스트 분할 정보가 저장된 경로. 기본값은 'data'.
        dataset (str): 사용할 데이터셋의 이름. 예: 'referit', 'flickr'.
        transform (callable): 이미지에 적용할 변환(augmentation) 함수 또는 callable 객체.
        return_idx (bool): 데이터셋 인덱스를 반환할지 여부. 기본값은 False.
        split (str): 사용할 데이터셋의 분할. 'train', 'val', 'test' 등이 가능. 기본값은 'train'.
        max_query_len (int): BERT 입력으로 사용할 텍스트의 최대 길이. 기본값은 128.
        lstm (bool): 텍스트를 LSTM을 사용할지 여부. 기본값은 False.
        bert_model (str): 사용할 BERT 모델 이름. 기본값은 'bert-base-uncased'.
        sample_size (int): 랜덤으로 샘플링할 데이터 개수. 기본값은 10.

    Usage Example:
        dataset = SupportGroundingDataset(
            data_root="/path/to/data", 
            split_root="/path/to/splits", 
            dataset="referit", 
            transform=some_transform, 
            sample_size=10
        )
        
        # 랜덤하게 샘플링된 데이터셋을 로드
        samples = dataset.random_sample()

    Returns:
        img (PIL.Image): 로드된 이미지.
        input_ids (list of int): BERT에 입력될 토큰 ID.
        input_mask (list of int): BERT 입력 마스크.
        bbox (torch.Tensor): 이미지 내 바운딩 박스 좌표.
    """
    def __init__(self, data_root, split_root='data', dataset='referit', 
                 transform=None, return_idx=False, split='train', 
                 max_query_len=128, lstm=False, bert_model='bert-base-uncased', sample_size=10):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.split = split
        self.sample_size = sample_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        # 데이터셋 로드 로직
        if not self.exists_dataset():
            raise FileNotFoundError(f"Dataset not found in {self.dataset_root}")

        dataset_path = osp.join(self.split_root, self.dataset)
        self.images = torch.load(osp.join(dataset_path, f"{self.dataset}_{self.split}.pth"))

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        img_file, bbox, phrase = self.images[idx] if self.dataset == 'flickr' else self.images[idx][:3]
        bbox = np.array(bbox, dtype=int)
        bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        
        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        bbox = torch.tensor(bbox).float()

        return img, phrase, bbox

    def parse_and_tokenize(self, phrase, idx):
        """주어진 문장을 파싱하고 BERT 입력 형식으로 변환합니다."""
        # 파싱
        examples = read_examples(phrase, unique_id=idx)

        # BERT 입력으로 변환
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer
        )

        # 첫 번째 feature를 사용 (한 문장만 처리하므로)
        input_ids = features[0].input_ids
        input_mask = features[0].input_mask

        return input_ids, input_mask

    def random_sample(self):
        """이미지와 텍스트를 무작위로 샘플링하여 가져옵니다."""
        sampled_idxs = random.sample(range(len(self.images)), self.sample_size)
        samples = [self.pull_item(idx) for idx in sampled_idxs]
        return samples

    def __getitem__(self, idx):
        # 이미지, 텍스트, 박스 정보 불러오기
        img, phrase, bbox = self.pull_item(idx)

        # 텍스트를 파싱하고 BERT 입력으로 변환
        input_ids, input_mask = self.parse_and_tokenize(phrase.lower(), idx)

        # 트랜스폼 적용
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']

        return img, input_ids, input_mask, bbox

    def __len__(self):
        return len(self.images)