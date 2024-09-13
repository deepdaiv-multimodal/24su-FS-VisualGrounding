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
        
    def set_support_dataset(self, support_images, support_texts):
        """
        Sets the support dataset for the few-shot learning.
        """
        #TODO : support_images, support_texts를 이용하여 
        # visual_features, language_features를 계산하고,
        # pseudo embedding을 추가하여 multimodal prompts를 계산 
        # # (B, N, C, H, W) -> (N, B, C, H, W)
        # visual_features = self.visumodel(support_images)
        # # (N, B, C, H, W) -> (N, B, H*W, C)
        # visual_features = self.visu_proj(visual_features)
        # # (N, B, H*W, C) -> (H*W, B, C)
        # self.visual_prompts = visual_features.mean(dim=0)
        
        # language_features = self.textmodel(support_texts)
        # language_features = self.text_proj(language_features)
        # self.language_prompts = language_features.mean(dim=0)
        
        # concatenate visual and language prompts
        self.visual_prompts = torch.cat([self.visual_prompts, self.language_prompts], dim=0)
        
        # add pseudo embeddings
        if hasattr(self, 'pseudo_embedding'):
            pseudo_embeddings = self.pseudo_embedding(torch.arange(self.pseudo_num_classes))
            self.visual_prompts = torch.cat([self.visual_prompts, pseudo_embeddings], dim=0)

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
            if self.different_transformer:
                vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
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