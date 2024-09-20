import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

from typing import List

class CategorizedRefCOCO:
    def __init__(self, 
                 file_name: str, 
                 sentences: List[str], 
                 gt_bbox: List[float], 
                 category: str, 
                 pseudo_categories: List[str]):
        self.file_name = file_name
        self.sentences = sentences
        self.gt_bbox = gt_bbox
        self.category = category
        self.pseudo_categories = pseudo_categories

    def __str__(self):
        #print all the attributes line by line
        return '\n'.join([f'{key}: {value}' for key, value in self.__dict__.items()])
    

class FSGroundingDataset(Dataset):
    def __init__(self, data_dict, transform=None, num_support_per_class=5, num_support_categories=10):
        self.class_names = data_dict['names']  # Class names list
        self.num_classes = len(self.class_names)
        train_dir = data_dict['train']  # Directory for training images and labels
        self.img_dir = os.path.join(train_dir, 'images')
        self.label_dir = os.path.join(train_dir, 'labels')
        self.transform = transform
        self.num_support_per_class = num_support_per_class

        # List to store all objects across images
        self.all_objects = []
        self.class_objects = {i: [] for i in range(self.num_classes)}  # Objects grouped by class

        # Load data from image and label files
        self._load_data()

        # Assert that each class has enough support objects
        assert all(len(objs) >= num_support_per_class for objs in self.class_objects.values())

    def _load_data(self):
        """
        Load all objects from images and label files.
        """
        for img_file in sorted(os.listdir(self.img_dir)):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.label_dir, label_file)

                # Only process if the label file exists
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            class_id = int(line.split()[0])
                            if class_id < self.num_classes:
                                obj = {
                                    'img_file': img_file,
                                    'class_id': class_id,
                                    'bbox_line': line.strip(),  # The label file line for this object
                                    'object_idx': idx  # Object index within the image
                                }
                                self.all_objects.append(obj)
                                self.class_objects[class_id].append(obj)

    def _create_support_dict(self, obj):
        """
        Create a support dictionary for a given object, cropping the object region.
        """
        img_file = obj['img_file']
        bbox_line = obj['bbox_line']
        class_id = obj['class_id']

        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # Extract the bounding box coordinates from the YOLO format
        parts = bbox_line.split()
        bbox = list(map(float, parts[1:]))  # [x_center, y_center, width, height]

        # Convert YOLO bbox to image coordinates
        img_w, img_h = image.size
        x_center, y_center, width, height = bbox
        left = (x_center - width / 2) * img_w
        top = (y_center - height / 2) * img_h
        right = (x_center + width / 2) * img_w
        bottom = (y_center + height / 2) * img_h

        # Crop the bounding box region
        cropped_image = image.crop((left, top, right, bottom))

        # Apply any transformation if provided
        if self.transform:
            cropped_image = self.transform(cropped_image)

        support_dict = {
            'text': self.class_names[class_id],  # Class name as text
            'img': cropped_image  # Cropped object image (PIL Image)
        }

        return support_dict

    def __len__(self):
        return len(self.all_objects)

    def __getitem__(self, idx):
        obj = self.all_objects[idx]
        support_dict = self._create_support_dict(obj)
        return support_dict

class FewShotDataset(Dataset):
    def __init__(self, yaml_file, transforms=None, num_support_per_class=4, num_support_categories=10):
        """
        Few-shot 학습용 데이터셋 클래스.
        :param yaml_file: 데이터셋 구성을 정의한 yaml 파일 경로
        :param transforms: 이미지 변환을 위한 torchvision transforms
        :param num_support_per_class: 각 클래스당 지원 이미지를 몇 개 불러올지 지정
        """
        import yaml

        self.transforms = transforms
        self.num_support_per_class = num_support_per_class

        # YAML 파일에서 데이터셋 정보를 읽어옴
        with open(yaml_file, 'r') as f:
            data_dict = yaml.safe_load(f)

        self.class_names = data_dict['names']  # 클래스 이름 목록을 가져옴
        self.num_classes = len(self.class_names)
        train_dir = data_dict['train']  # 'train' 키를 사용하여 이미지 디렉토리 지정
        # img_dir : train_dir/images, label_dir : train_dir/labels
        self.img_dir = os.path.join(train_dir, 'images')
        self.label_dir = os.path.join(train_dir, 'labels')

        # 모든 객체 정보를 저장할 리스트 초기화
        self.all_objects = []  # 전체 객체 리스트
        self.class_objects = {i: [] for i in range(self.num_classes)}  # 클래스별 객체 리스트

        # 이미지와 라벨을 읽어 모든 객체를 수집
        self._load_data()
        
        # 각각의 객체 개수 > num_support_per_class 함
        assert all(len(objs) >= num_support_per_class for objs in self.class_objects.values())

    def _load_data(self):
        """
        이미지와 라벨 파일을 읽어와 모든 객체를 수집하는 함수.
        """
        for img_file in sorted(os.listdir(self.img_dir)):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.label_dir, label_file)

                # 라벨 파일이 존재하는 경우에만 처리
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for idx, line in enumerate(lines):
                            class_id = int(line.split()[0])
                            if class_id < self.num_classes:
                                obj = {
                                    'img_file': img_file,
                                    'class_id': class_id,
                                    'bbox_line': line.strip(),  # 라벨 파일의 해당 라인
                                    'object_idx': idx  # 객체 인덱스 (이미지 내에서)
                                }
                                self.all_objects.append(obj)
                                self.class_objects[class_id].append(obj)

    def __len__(self):
        # 전체 객체 수를 반환
        return len(self.all_objects)

    def _create_support_dict(self, obj):
        """
        객체 정보를 받아 지원 이미지 딕셔너리를 생성하는 함수.
        """
        img_file = obj['img_file']
        bbox_line = obj['bbox_line']
        class_id = obj['class_id']

        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # 바운딩 박스 좌표 추출
        parts = bbox_line.split()
        bbox = list(map(float, parts[1:]))  # [x_center, y_center, width, height]

        # YOLO 형식의 바운딩 박스를 이미지 좌표로 변환
        img_w, img_h = image.size
        x_center, y_center, width, height = bbox
        left = (x_center - width / 2) * img_w
        top = (y_center - height / 2) * img_h
        right = (x_center + width / 2) * img_w
        bottom = (y_center + height / 2) * img_h

        # 바운딩 박스 영역만 잘라냄
        cropped_image = image.crop((left, top, right, bottom))

        support_dict = {
            'text': self.class_names[class_id],  # 클래스 이름
            'img': cropped_image  # 잘라낸 객체 이미지 (PIL 객체)
        }

        return support_dict

    def _get_support_images(self, class_id, query_obj):
        """
        모든 클래스에 대한 지원 이미지를 불러오는 함수.
        """
        support_dicts = []

        # 동일한 클래스에서 num_support_per_class개의 지원 이미지 선택
        same_class_objects = [obj for obj in self.class_objects[class_id] if obj != query_obj]
        num_same_class = self.num_support_per_class
        sampled_same_class = random.sample(same_class_objects, num_same_class)

        for obj in sampled_same_class:
            support_dicts.append(self._create_support_dict(obj))

        # 다른 클래스에서 num_support_per_class개의 지원 이미지 선택
        for other_class_id in range(self.num_classes):
            if other_class_id == class_id:
                continue
            other_class_objects = self.class_objects[other_class_id]
            num_other_class = self.num_support_per_class
            if num_other_class > 0:
                sampled_other_class = random.sample(other_class_objects, num_other_class)
                for obj in sampled_other_class:
                    support_dicts.append(self._create_support_dict(obj))

        return support_dicts

    def __getitem__(self, idx):
        """
        인덱스에 따라 쿼리 이미지와 지원 세트를 반환하는 메서드.
        """
        query_obj = self.all_objects[idx]
        img_file = query_obj['img_file']
        class_id = query_obj['class_id']
        bbox_line = query_obj['bbox_line']

        # 쿼리 이미지 로드
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # 쿼리 객체의 바운딩 박스 추출
        parts = bbox_line.split()
        bbox = list(map(float, parts[1:]))

        # 바운딩 박스 좌표 변환 (YOLO 형식 -> 이미지 좌표)
        img_w, img_h = image.size
        x_center, y_center, width, height = bbox
        left = (x_center - width / 2) * img_w
        top = (y_center - height / 2) * img_h
        right = (x_center + width / 2) * img_w
        bottom = (y_center + height / 2) * img_h

        # 쿼리 객체의 바운딩 박스 영역을 이미지에 표시하거나 사용할 수 있음

        # 쿼리 이미지의 전체 라벨 로드
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    class_id_label = int(parts[0])
                    bbox_label = list(map(float, parts[1:]))
                    boxes.append([class_id_label] + bbox_label)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 5), dtype=torch.float32)

        # 쿼리 이미지 정보 생성
        query_dict = {
            'text': self.class_names[class_id],  # 클래스 이름
            'img': image,  # 이미지 (PIL 객체)
            'box': boxes   # 바운딩 박스 (좌표)
        }

        # transforms가 정의되어 있으면 쿼리 이미지에도 적용
        if self.transforms:
            query_dict = self.transforms(query_dict)

        # 해당 객체에 대한 지원 이미지와 레이블 가져오기
        support_dicts = self._get_support_images(class_id, query_obj)

        # 쿼리 이미지와 지원 이미지를 반환
        return query_dict, support_dicts