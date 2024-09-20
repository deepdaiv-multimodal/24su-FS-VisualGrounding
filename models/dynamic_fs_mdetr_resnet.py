import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .dynamic_mdetr_resnet import DynamicMDETR

class DynamicFSMDETR(DynamicMDETR):
    def __init__(self, args):
        super(DynamicFSMDETR, self).__init__(args)
        self.pseudo_num_classes = args.pseudo_num_classes
        self.hidden_dim = args.hidden_dim
        if args.pseudo_embedding:
            # PseudoEmbedding 클래스를 사용하여 learnable한 pseudo embedding 초기화
            assert args.hidden_dim == args.vl_hidden_dim, "Hidden dimension of the model and pseudo embedding should be the same."
            self._init_pseudo_embedding()
            self._freeze_model_parameters()

    def _freeze_model_parameters(self):
        """
        Freezes all parameters except for query and PseudoEmbedding.
        """
        for name, param in self.named_parameters():
            if ('pseudo_embedding' not in name) and ('init_sampling_feature') not in name and 'update_sampling_queries' not in name:
                param.requires_grad = False
    
    def _init_pseudo_embedding(self):
        """
        Initializes the pseudo embedding.
        """
        self.pseudo_embedding = PseudoEmbedding(self.pseudo_num_classes, self.hidden_dim)
        
    def set_support_dataset(self, tem_imgs, tem_txts, category, tem_categories):
        """
        Sets the support dataset for the few-shot learning.
        """
        # Category -> Index
        category_idx = torch.tensor([self.category_to_idx[cat] for cat in category], device=tem_imgs.tensors.device)
        
        
        # 1. Visual Encoder
        B, N, _, _, _ = tem_imgs.size()
        # (B, N, C, H, W) -> (B * N, C, H, W)
        tem_imgs = tem_imgs.view(-1, *tem_imgs.shape[2:])
        # (B * N, C, H, W) -> (B * N, N_v, visual_num_channels)
        # 여기서 Batch에 대해 반복문 안 돌리고 그냥 모델에 넣어도 될 듯?
        tem_out, tem_visu_pos = self.visumodel(tem_imgs)
        tem_visu_src, tem_visu_mask = tem_out
        # (B * N, N_v, visual_num_channels) -> (B * N, N_v, hidden_dim)
        tem_visu_src = self.visu_proj(tem_visu_src)
        
        # (B * N, N_v, hidden_dim) -> (B, N, N_v, hidden_dim)
        visual_src = tem_visu_src.view(B, N, *tem_visu_src.shape[1:])
        # Average pooling
        # (B, N, N_v, hidden_dim) -> (B, N, hidden_dim)
        visual_src = visual_src.mean(dim=2)
        
        
        # 2. Language Encoder
        B, N, L = tem_txts.size()
        # (B, N, L) -> (B * N, L)
        tem_txts = tem_txts.view(-1, *tem_txts.shape[2:])
        
        # (B * N, L) -> (B * N, L, text_num_channels)
        tem_text_fea = self.textmodel(tem_txts)
        tem_text_src, tem_text_mask = tem_text_fea.decompose()
        
        # (B * N, L, text_num_channels) -> (B * N, L, hidden_dim)
        tem_text_src = self.text_proj(tem_text_src)
        # (B * N, L, hidden_dim) -> (B, N, L, hidden_dim)
        text_src = tem_text_src.view(B, N, *tem_text_src.shape[1:])
        # Average pooling
        # (B, N, L, hidden_dim) -> (B, N, hidden_dim)
        text_src = text_src.mean(dim=2)
        
        
        # concatenate visual and language prompts
        # (B, N, hidden_dim) -> (B, N, 1, hidden_dim)
        visual_src, text_src = visual_src.unsqueeze(2), text_src.unsqueeze(2)
        
        # (B, N, 1, hidden_dim) -> (B, N, 2, hidden_dim)
        vl_src = torch.cat([visual_src, text_src], dim=2)
        
        # Average pooling to dim 3
        # (B, N, 2, hidden_dim) -> (B, N, hidden_dim)
        vl_src = vl_src.mean(dim=2)
        
        assert vl_src.shape == (B, N, self.hidden_dim), \
    f"vl_src shape should be {(B, N, self.hidden_dim)}, but got {vl_src.shape}"
        
        # Add pseudo embeddings
        # Same Category -> Same Pseudo embedding
        # the pseudo embedding should be learnable.
        # Pseudo embedding is chosen randomly
        pseudo_indexes = category_idx
        # random matching
        pseudo_indexes = torch.randint(0, self.pseudo_num_classes, (B,), device=tem_imgs.tensors.device)
        pseudo_embeddings = self.pseudo_embedding(pseudo_indexes)
        
        # (B, N, hidden_dim) -> (B, N, hidden_dim)
        multimodal_prompt = vl_src + pseudo_embeddings
        
        self.multimodal_prompt = multimodal_prompt
        

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # 1. Feature Encoder

        # 1.1 Visual Encoder
        # visual backbone
        out, visu_pos = self.visumodel(img_data)
        visu_mask, visu_src = out # (B, H*W), (H*W, B, channel)
        visu_src = self.visu_proj(visu_src)  # (H*W, B, channel)

        # 1.2 Language Encoder
        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        # text_src: (bs, max_len, channel)
        text_mask = text_mask.flatten(1)  # (B, max_len)
        text_src = self.text_proj(text_src).permute(1, 0, 2)  # (max_len, B, channel)

        # 1.3 Concat visual features, language features, and visual prompts
        visual_prompts = self.visual_prompts.unsqueeze(1).repeat(1, bs, 1)  # Shape: (num_templates, B, hidden_dim)
        vl_src = torch.cat([visual_prompts, visu_src, text_src], dim=0)  # Concat visual prompts with other features
        vl_mask = torch.cat([torch.zeros(visual_prompts.size(0), bs).to(text_mask.device), visu_mask, text_mask], dim=1)  # Concat masks
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # 2. Multimodal Transformer
        # 2.1 Multimodal Transformer Encoder
        if self.vl_encoder is not None:
            vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
        else:
            vl_feat = vl_src

        # 2.2 Split back to visual prompts, visual features, and language features
        num_prompts = self.visual_prompts.size(0)
        visu_feat = vl_feat[:self.num_visu_token] # (H*W, B, channel)
        language_feat = vl_feat[num_prompts + self.num_visu_token:]  # (max_len, B, channel)
        v_pos = vl_pos[:self.num_visu_token]
        l_pos = vl_pos[self.num_visu_token:]

        # 2.3 Dynamic Multimodal Transformer Decoder
        # Initialize sampling query and reference point for the first features sampling
        sampling_query = self.init_sampling_feature.weight.repeat(bs, 1)
        reference_point = self.init_reference_point.weight.repeat(bs, 1)
        pred_box = None

        for i in range(0, self.stages):
            # 2D adaptive sampling
            sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, visu_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

            # Text guided decoding with one-layer transformer encoder-decoder

            # Language_feat : (max_len, B, hidden_dim)
            # multimodal_prompt : (B, N, hidden_dim)
            # concat -> (max_len + N, B, hidden_dim)
            '''
            Concat Multimodal Prompt & Language Feature Here !!!
            '''
            language_feat_with_prompt = torch.cat([language_feat, self.multimodal_prompt.unsqueeze(0).repeat(language_feat.size(0), 1, 1)], dim=0)
            
            if self.different_transformer:
                vg_hs = self.vl_transformer[i](sampled_features, None, language_feat_with_prompt, pe, text_mask, l_pos)[0]
            else:
                vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

            # Prediction Head
            language_feat = vg_hs[0]

            text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
            text_select_num = text_select.sum(dim=1)  # (bs, 1)

            # new language queries
            vg_hs = (text_select * vg_hs[0].permute(1,0,2)).sum(dim=1) / text_select_num  # (bs, channel)

            pred_box = self.bbox_embed(vg_hs).sigmoid()

            # Update reference point and sampling query
            reference_point = pred_box[:, :2]
            sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

        return pred_box

            
class PseudoEmbedding(nn.Module):
    def __init__(self, pseudo_num_classes, embedding_dim):
        """
        Initializes the PseudoEmbedding module.

        Args:
            pseudo_num_classes (int): Number of pseudo-classes.
            embedding_dim (int): Dimension of the embeddings.
        """
        super(PseudoEmbedding, self).__init__()
        # Initialize pseudo-class embeddings from a normal distribution
        self.embeddings = nn.Parameter(torch.empty(pseudo_num_classes, embedding_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=1.0)  # Normal distribution initialization

    def forward(self, indexes):
        """
        Forward method to retrieve pseudo-class embeddings.

        Args:
            indexes (torch.Tensor): Indexes of the pseudo-classes.

        Returns:
            torch.Tensor: Corresponding pseudo-class embeddings.
        """
        return self.embeddings[indexes]