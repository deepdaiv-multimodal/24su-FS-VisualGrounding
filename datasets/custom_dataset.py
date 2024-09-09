import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from pycocotools.coco import COCO
import datasets.transforms as T

class FewShotGroundingDataset(VisionDataset):
    def __init__(self, root, annFiles, img_folder, support_folder, transforms=None, support_transforms=None,
                 return_masks=False, with_support=False, cache_mode=False, local_rank=0, local_size=1):
        """
        Few-Shot Grounding Dataset integrating both query and support datasets for few-shot learning scenarios.

        Args:
            root (str): Root directory for datasets.
            annFiles (list): Annotation files.
            img_folder (str): Image folder for query images.
            support_folder (str): Image folder for support images.
            transforms (callable, optional): Transformations to apply on query images.
            support_transforms (callable, optional): Transformations to apply on support images.
            return_masks (bool): Whether to return masks for segmentation tasks.
            with_support (bool): Whether to include support images in the dataset.
            cache_mode (bool): Whether to cache dataset loading.
            local_rank (int): Local rank for distributed processing.
            local_size (int): Local size for distributed processing.
        """
        super(FewShotGroundingDataset, self).__init__(root)
        self.coco = COCO(annFiles)  # Initialize COCO
        self.img_folder = img_folder
        self.support_folder = support_folder
        self.transforms = transforms or self.make_transforms()
        self.support_transforms = support_transforms or self.make_support_transforms()
        self.with_support = with_support
        self.return_masks = return_masks
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size

        # If with_support, initialize support dataset
        if self.with_support:
            self.support_dataset = SupportDataset(root=support_folder, annFiles=annFiles,
                                                  transforms=self.support_transforms)

    def __getitem__(self, index):
        """
        Retrieve a query image and its corresponding text. If `with_support` is True,
        retrieve the corresponding support image and its ground text as well.
        """
        # Load query image and its corresponding text
        img_id = self.coco.getImgIds()[index]
        img_data = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_data['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Retrieve associated ground text for the query image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        ground_text = [ann['caption'] for ann in anns]

        if self.transforms:
            img = self.transforms(img)

        if self.with_support:
            # Retrieve a support image and its corresponding ground text
            support_img, support_text = self.support_dataset[index] if len(self.support_dataset) > 0 else (None, None)
            return img, ground_text, support_img, support_text

        return img, ground_text

    def __len__(self):
        return len(self.coco.getImgIds())

    @staticmethod
    def make_transforms():
        """Transformations for query images."""
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(p=0.3333),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1152),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1152),
                ])
            ),
            normalize,
        ])

    @staticmethod
    def make_support_transforms():
        """Transformations for support images."""
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

class SupportDataset(VisionDataset):
    """
    Support Dataset class, specifically used during the inference stage.
    """
    def __init__(self, root, annFiles, transforms=None):
        super(SupportDataset, self).__init__(root)
        self.coco = COCO(annFiles)
        self.transforms = transforms or FewShotGroundingDataset.make_support_transforms()

    def __getitem__(self, index):
        """
        Retrieve a support image and its corresponding ground text.
        """
        img_id = self.coco.getImgIds()[index]
        img_data = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_data['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Retrieve associated ground text for the support image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        support_text = [ann['caption'] for ann in anns]

        if self.transforms:
            img = self.transforms(img)

        return img, support_text

    def __len__(self):
        return len(self.coco.getImgIds())