import numpy as np
import torch
from torch.nn.functional import softmax

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rule 10 from paper
def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_transformer_att(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # R_q_i generated from last layer
        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn_cam().detach()
        grad_q_i = decoder_last.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.self_attn.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.self_attn.get_attn_cam().detach()
            else:
                cam = blk.self_attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)

    def handle_co_attn_self_query(self, block):
        grad = block.self_attn.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.self_attn.get_attn_cam().detach()
        else:
            cam = block.self_attn.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add

    def handle_co_attn_query(self, block):
        if self.use_lrp:
            cam_q_i = block.multihead_attn.get_attn_cam().detach()
        else:
            cam_q_i = block.multihead_attn.get_attn().detach()
        grad_q_i = block.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
                                               apply_normalization=self.normalize_self_attention,
                                               apply_self_in_rule_10=self.apply_self_in_rule_10)

    def generate_ours(self, img, target_index, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        outputs = self.model(img)
        outputs = outputs['pred_logits']
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs[0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs).to(outputs.device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        if use_lrp:
            self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # image self attention in the encoder
        self.handle_self_attention_image(encoder_blocks)

        # decoder self attention of queries followd by multi-modal attention
        for blk in decoder_blocks:
            # decoder self attention
            self.handle_co_attn_self_query(blk)

            # encoder decoder attention
            self.handle_co_attn_query(blk)
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:,target_index, :].unsqueeze_(0).detach()
        return aggregated

    def generate_partial_lrp(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()

        self.model.relprop(one_hot_vector, **kwargs)

        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn_cam().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        # normalize to get non-negative cams
        self.R_q_i = (self.R_q_i - self.R_q_i.min()) / (self.R_q_i.max() - self.R_q_i.min())
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def generate_raw_attn(self, img, target_index):
        outputs = self.model(img)

        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def generate_rollout(self, img, target_index):
        outputs = self.model(img)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        cams_image = []
        cams_queries = []
        # image self attention in the encoder
        for blk in encoder_blocks:
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_image.append(cam)

        # decoder self attention of queries
        for blk in decoder_blocks:
            # decoder self attention
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_queries.append(cam)

        # rollout for self-attention values
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_q_q = compute_rollout_attention(cams_queries)

        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = torch.matmul(self.R_q_q.t(), torch.matmul(cam_q_i, self.R_i_i))
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, img, target_index, index=None):
        outputs = self.model(img)

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)


        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        grad_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn_gradients().detach()
        cam_q_i = self.gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated




class GeneratorAlbationNoAgg:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.self_attn.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.self_attn.get_attn_cam().detach()
            else:
                cam = blk.self_attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i = torch.matmul(cam, self.R_i_i)

    def handle_co_attn_self_query(self, block):
        grad = block.self_attn.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.self_attn.get_attn_cam().detach()
        else:
            cam = block.self_attn.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q = R_q_q_add
        self.R_q_i = R_q_i_add

    def handle_co_attn_query(self, block):
        if self.use_lrp:
            cam_q_i = block.multihead_attn.get_attn_cam().detach()
        else:
            cam_q_i = block.multihead_attn.get_attn().detach()
        grad_q_i = block.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
                                               apply_normalization=self.normalize_self_attention,
                                               apply_self_in_rule_10=self.apply_self_in_rule_10)

    def generate_ours_abl(self, img, target_index, index=None, use_lrp=False, normalize_self_attention=False, apply_self_in_rule_10=True):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        outputs = self.model(img)
        outputs = outputs['pred_logits']
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs[0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs).to(outputs.device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        if use_lrp:
            self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # image self attention in the encoder
        self.handle_self_attention_image(encoder_blocks)

        # decoder self attention of queries followd by multi-modal attention
        for blk in decoder_blocks:
            # decoder self attention
            self.handle_co_attn_self_query(blk)

            # encoder decoder attention
            self.handle_co_attn_query(blk)
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:,target_index, :].unsqueeze_(0).detach()
        return aggregated
