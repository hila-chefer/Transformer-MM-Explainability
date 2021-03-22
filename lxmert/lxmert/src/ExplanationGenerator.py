import numpy as np
import torch
import copy

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
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

# rules 10 + 11 from paper
def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t += R_t_t_add
            self.R_t_i += R_t_i_add

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i += R_i_i_add
            self.R_i_t += R_i_t_add

    def handle_co_attn_self_lang(self, block):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.lang_self_att.self.get_attn_cam().detach()
        else:
            cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add

    def handle_co_attn_self_image(self, block):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.visn_self_att.self.get_attn_cam().detach()
        else:
            cam = block.visn_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add

    def handle_co_attn_lang(self, block):
        if self.use_lrp:
            cam_t_i = block.visual_attention.att.get_attn_cam().detach()
        else:
            cam_t_i = block.visual_attention.att.get_attn().detach()
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block):
        if self.use_lrp:
            cam_i_t = block.visual_attention_copy.att.get_attn_cam().detach()
        else:
            cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_i_t_addition, R_i_i_addition

    def generate_ours(self, input, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True, method_name="ours"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        if self.use_lrp:
            model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk)

            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            # language self attention
            self.handle_co_attn_self_lang(blk)

            # image self attention
            self.handle_co_attn_self_image(blk)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
        self.R_t_i += R_t_i_addition
        self.R_t_t += R_t_t_addition

        # language self attention
        self.handle_co_attn_self_lang(blk)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i



class GeneratorOursAblationNoAggregation:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t = R_t_t_add
            self.R_t_i = R_t_i_add

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i = R_i_i_add
            self.R_i_t = R_i_t_add

    def handle_co_attn_self_lang(self, block):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.lang_self_att.self.get_attn_cam().detach()
        else:
            cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t = R_t_t_add
        self.R_t_i = R_t_i_add

    def handle_co_attn_self_image(self, block):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.visn_self_att.self.get_attn_cam().detach()
        else:
            cam = block.visn_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i = R_i_i_add
        self.R_i_t = R_i_t_add

    def handle_co_attn_lang(self, block):
        if self.use_lrp:
            cam_t_i = block.visual_attention.att.get_attn_cam().detach()
        else:
            cam_t_i = block.visual_attention.att.get_attn().detach()
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i, apply_normalization=self.normalize_self_attention)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block):
        if self.use_lrp:
            cam_i_t = block.visual_attention_copy.att.get_attn_cam().detach()
        else:
            cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t, apply_normalization=self.normalize_self_attention)
        return R_i_t_addition, R_i_i_addition

    def generate_ours_no_agg(self, input, index=None, use_lrp=False, normalize_self_attention=True, method_name="ours_no_agg"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        if self.use_lrp:
            model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk)

            self.R_t_i = R_t_i_addition
            self.R_t_t = R_t_t_addition
            self.R_i_t = R_i_t_addition
            self.R_i_i = R_i_i_addition

            # language self attention
            self.handle_co_attn_self_lang(blk)

            # image self attention
            self.handle_co_attn_self_image(blk)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
        self.R_t_i = R_t_i_addition
        self.R_t_t = R_t_t_addition

        # language self attention
        self.handle_co_attn_self_lang(blk)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i


class GeneratorBaselines:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def generate_transformer_attr(self, input, index=None, method_name="transformer_attr"):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)


        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break

            # language self attention
            grad = blk.lang_self_att.self.get_attn_gradients().detach()
            cam = blk.lang_self_att.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

            # image self attention
            grad = blk.visn_self_att.self.get_attn_gradients().detach()
            cam = blk.visn_self_att.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn_cam().detach()
        grad_t_i = blk.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        grad = blk.lang_self_att.self.get_attn_gradients().detach()
        cam = blk.lang_self_att.self.get_attn_cam().detach()
        cam = avg_heads(cam, grad)
        self.R_t_t += torch.matmul(cam, self.R_t_t)

        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i

    def generate_partial_lrp(self, input, index=None, method_name="partial_lrp"):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.zeros(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.zeros(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn_cam().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = cam_t_i

        # language self attention
        cam = blk.lang_self_att.self.get_attn_cam().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam

        # normalize to get non-negative cams
        self.R_t_t = (self.R_t_t - self.R_t_t.min()) / (self.R_t_t.max() - self.R_t_t.min())
        self.R_t_i = (self.R_t_i - self.R_t_i.min()) / (self.R_t_i.max() - self.R_t_i.min())
        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def generate_raw_attn(self, input, method_name="raw_attention"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.zeros(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.zeros(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        cam = blk.lang_self_att.self.get_attn().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam

        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, input, index=None, method_name="gradcam"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        grad_t_i = blk.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = self.gradcam(cam_t_i, grad_t_i)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        grad = blk.lang_self_att.self.get_attn_gradients().detach()
        cam = blk.lang_self_att.self.get_attn().detach()
        self.R_t_t = self.gradcam(cam, grad)

        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def generate_rollout(self, input, method_name="rollout"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        cams_text = []
        cams_image = []
        # language self attention
        blocks = model.lxmert.encoder.layer
        for blk in blocks:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)


        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break

            # language self attention
            cam = blk.lang_self_att.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

            # image self attention
            cam = blk.visn_self_att.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_t = compute_rollout_attention(copy.deepcopy(cams_text))
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        # language self attention
        cam = blk.lang_self_att.self.get_attn().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        cams_text.append(cam)

        self.R_t_t = compute_rollout_attention(cams_text)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i