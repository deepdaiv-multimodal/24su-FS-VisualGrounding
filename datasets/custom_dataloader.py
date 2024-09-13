import torch
from torchvision.transforms import ToTensor

def collate_fn(batch):
    """
    커스텀 collate 함수는 각 배치의 쿼리와 지원 이미지를 동일한 크기로 리사이즈하고,
    바운딩 박스와 클래스 레이블을 텐서로 변환하여 묶습니다.
    """
    to_tensor = ToTensor()  # PIL 이미지를 텐서로 변환하는 함수
    resize_size = (640, 640)  # 원하는 리사이즈 크기
    
    # Query images and labels
    query_images = []
    query_boxes = []
    query_texts = []  # 클래스 이름 저장
    
    # Support images and labels for each query
    support_images_all = []
    support_texts_all = []  # 클래스 이름 저장

    for sample in batch:
        query_dict, support_dicts = sample
        
        # Query images, labels, and texts
        query_img = query_dict['img']
        if not isinstance(query_img, torch.Tensor):
            query_img = query_img.resize(resize_size)
            query_img = to_tensor(query_img)  # PIL 이미지 -> Tensor 변환
        query_images.append(query_img)
        query_boxes.append(query_dict['box'])
        query_texts.append(query_dict['text'])  # 클래스 이름 추가
        
        # Support images and texts (no bounding boxes)
        support_images_batch = []
        support_texts_batch = []
        
        for support in support_dicts:
            support_img = support['img']
            if not isinstance(support_img, torch.Tensor):
                support_img = support_img.resize(resize_size)
                support_img = to_tensor(support_img)  # PIL 이미지 -> Tensor 변환
            support_images_batch.append(support_img)
            support_texts_batch.append(support['text'])  # 클래스 이름 추가

        support_images_all.append(support_images_batch)
        support_texts_all.append(support_texts_batch)

    # Convert lists to tensors
    query_images = torch.stack(query_images)  # Shape: (B, C, H, W)
    
    # Padding boxes and converting to tensors
    max_boxes = max([box.shape[0] for box in query_boxes])
    query_boxes_padded = [torch.cat([box, torch.zeros(max_boxes - box.shape[0], box.shape[1])], dim=0) for box in query_boxes]
    query_boxes_padded = torch.stack(query_boxes_padded)
    
    # Support set processing (support boxes are not needed)
    support_images_tensors = []
    for support_images in support_images_all:
        support_images_tensors.append(torch.stack(support_images))

    # Stacking all support images for the batch
    support_images_tensors = torch.stack(support_images_tensors)

    # Returning as a dictionary with texts included
    return {
        'query_images': query_images,                # 쿼리 이미지들
        'query_boxes': query_boxes_padded,           # 쿼리 이미지의 바운딩 박스
        'query_texts': query_texts,                  # 쿼리 이미지의 클래스 이름들
        'support_images': support_images_tensors,    # 지원 이미지들
        'support_texts': support_texts_all           # 지원 이미지의 클래스 이름들
    }