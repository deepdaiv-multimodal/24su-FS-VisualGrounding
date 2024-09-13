import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist

import util.misc as utils
from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from models.clip import clip

from datasets import build_my_dataset, collate_fn
from models import build_my_model


def train_one_epoch(args,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0):
    """
    한 에포크 동안 모델을 학습하는 함수.
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50  # 50번의 배치마다 로그를 출력

    # 데이터셋 로드 및 DataLoader 생성
    dataset = build_my_dataset(split='train', args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    for batch_idx, batch in enumerate(dataloader):
        # 쿼리 및 지원 데이터를 디바이스로 이동
        query_images = batch['query_images'].to(device)
        query_boxes = batch['query_boxes'].to(device)
        query_texts = batch['query_texts']
        support_images = batch['support_images'].to(device)
        support_texts = batch['support_texts']

        # support images와 support text 각각 Encoding
        support_images = model.set_support_dataset(support_images, support_texts)
        # # 가중치가 적용된 손실 계산
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # # 손실 역전파
        # optimizer.zero_grad()
        # losses.backward()

        # # 그라디언트 클리핑 (max_norm이 0보다 큰 경우)
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()

        # 로그 기록
        # metric_logger.update(loss=losses.item(), **loss_dict)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if max_norm > 0:
        #     grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #     metric_logger.update(grad_norm=grad_total_norm)

    # 에포크가 끝날 때 메트릭 출력
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch
        if args.model_type == "ResNet":
            batch_size = img_data.tensors.size(0)
        else:
            batch_size = img_data.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        else:
            text_data = clip.tokenize(text_data).to(device)
        target = target.to(device)

        pred_boxes = model(img_data, text_data)

        loss_dict = loss_utils.trans_vg_loss(pred_boxes, target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)
    return stats

@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        else:
            text_data = clip.tokenize(text_data).to(device)
        target = target.to(device)
        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy

# For testing the training loop

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--lr_finetune', default=1e-5, type=float)


    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='DynamicMDETR',
                        help="Name of model to be exploited.")
    parser.add_argument('--model_type', type=str, default='ResNet', choices=('ResNet', 'CLIP'),
                        help="Name of model to be exploited.")
    
    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # new
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')
    parser.add_argument('--in_points', default=32, type=int)
    parser.add_argument('--stages', default=6, type=int)

    # FS-MDETR parameters
    parser.add_argument('--pseudo_embedding', action='store_true',
                        help="If true, use pseudo-class embeddings")
    parser.add_argument('--pseudo_num_classes', default=100, type=int,
                        help="Number of pseudo-classes")

    # Dataset parameters
    parser.add_argument('--yaml_file', type=str, default='ln_data/Teenieping/data.yaml',
                        help='path to yaml file')
    parser.add_argument('--data_root', type=str, default='/data1/shifengyuan/visual_grounding',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--num_support_per_class', default=3, type=int,
                        help='number of support images per class')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default=None, type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--uniform_grid', default=False, type=bool)
    parser.add_argument('--uniform_learnable', default=False, type=bool)
    parser.add_argument('--different_transformer', default=False, type=bool)
    parser.add_argument('--vl_fusion_enc_layers', default=3, type=int)
    return parser

if __name__ == "__main__":
    # 모델과 기타 구성 요소 초기화 (예시)
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 생성
    dataset = build_my_dataset(split='train', args=args)

    # DataLoader 생성
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 생성 (예시 모델)
    model = build_my_model(args).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.CrossEntropyLoss()  # 예시: CrossEntropyLoss
    optimizer = Adam(model.parameters(), lr=args.lr)  # 예시: Adam optimizer

    # 학습 시작
    for epoch in range(1, 11):  # 10 에포크 동안 학습
        train_one_epoch(args, model, criterion, dataloader, optimizer, device, epoch)