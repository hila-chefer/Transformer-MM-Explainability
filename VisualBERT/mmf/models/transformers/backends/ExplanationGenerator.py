import numpy as np
import torch
import cv2

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class SelfAttentionGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_transformer_att(self, input, index=None, start_layer=0, save_visualization=False, save_visualization_per_token=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        cams = []
        blocks = self.model.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0))
        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = rollout[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='trans_att')

        if save_visualization_per_token:
            for token in range(1,cls_index+1):
                token_relevancies = rollout[:, token]
                token_relevancies[:, token] = 0
                save_visual_results(input, token_relevancies, method_name='trans_att', expl_path='per_token', suffix=str(token))
        return cls_per_token_score

    def generate_ours(self, input, index=None, save_visualization=False, save_visualization_per_token=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        blocks = self.model.model.bert.encoder.layer
        num_tokens = blocks[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.self.get_attn().device)
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = R[cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='ours')

        if save_visualization_per_token:
            for token in range(1,cls_index+1):
                token_relevancies = R[:, token]
                token_relevancies[:, token] = 0
                save_visual_results(input, token_relevancies, method_name='ours', expl_path='per_token', suffix=str(token))
        return cls_per_token_score

    def generate_partial_lrp(self, input, index=None, save_visualization=False):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1

        self.model.relprop(torch.tensor(one_hot).to(output.device), **kwargs)

        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        # cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)
        cam = cam.mean(dim=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='partial_lrp')
        return cls_per_token_score

    # def generate_full_lrp(self, input_ids, attention_mask,
    #                  index=None):
    #     output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    #     kwargs = {"alpha": 1}
    #
    #     if index == None:
    #         index = np.argmax(output.cpu().data.numpy(), axis=-1)
    #
    #     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #     one_hot[0, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * output)
    #
    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
    #     cam = cam.sum(dim=2)
    #     cam[:, 0] = 0
    #     return cam

    def generate_raw_attn(self, input, save_visualization=False):
        output = self.model(input)['scores']
        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='raw_attn')
        return cls_per_token_score

    def generate_rollout(self, input, start_layer=0, save_visualization=False):
        output = self.model(input)['scores']
        blocks = self.model.model.bert.encoder.layer
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = rollout[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='rollout')
        return cls_per_token_score

    def generate_attn_gradcam(self, input, index=None, save_visualization=False):
        output = self.model(input)['scores']

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cam = self.model.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        grad = self.model.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()[0]

        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = cam[0, cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name='gradcam')
        return cls_per_token_score

# class MultiAttentionGenerator:
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()
#
#     def generate_LRP(self, input, index=None, start_layer=0, save_visualization=False, save_visualization_per_token=False):
#         output = self.model(input)['scores']
#         kwargs = {"alpha": 1}
#
#         if index == None:
#             index = np.argmax(output.cpu().data.numpy(), axis=-1)
#
#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0, index] = 1
#         one_hot_vector = one_hot
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         one_hot = torch.sum(one_hot.cuda() * output)
#
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#
#         # self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)
#
#         cams = []
#         blocks = self.model.model.bert.encoder.layer
#         for blk in blocks:
#             grad = blk.attention.self.get_attn_gradients()
#             cam = blk.attention.self.get_attn_cam()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#             cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             cams.append(cam.unsqueeze(0))
#         rollout = compute_rollout_attention(cams, start_layer=start_layer)
#         input_mask = input['input_mask']
#         cls_index = input_mask.sum(1) - 2
#         cls_per_token_score = rollout[0, cls_index]
#         cls_per_token_score[:, cls_index] = 0
#
#         if save_visualization:
#             save_visual_results(input, cls_per_token_score, method_name='ours')
#
#         if save_visualization_per_token:
#             for token in range(1,cls_index+1):
#                 token_relevancies = rollout[:, token]
#                 token_relevancies[:, token] = 0
#                 save_visual_results(input, token_relevancies, method_name='ours', expl_path='per_token', suffix=str(token))
#         return cls_per_token_score
