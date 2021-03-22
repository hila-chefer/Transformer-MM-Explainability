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

def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

def generate(text_list, attention_list, latex_file, color='red'):
    attention_list = attention_list[:len(text_list)]
    if attention_list.max() == attention_list.min():
        attention_list = torch.zeros_like(attention_list)
    else:
        cls_position = attention_list.shape[0]-2
        attention_list[cls_position] = attention_list.max()
        attention_list[cls_position] = attention_list.min()
        attention_list = 100 * (attention_list - attention_list.min()) / (attention_list.max() - attention_list.min())
    attention_list[attention_list < 1] = 0
    attention_list = attention_list.tolist()
    text_list = [text_list[i].replace('$', '') for i in range(len(text_list))]
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth=150mm]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            if '\#\#' in text_list[idx]:
                token = text_list[idx].replace('\#\#', '')
                string += "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + token + "}"
            else:
                string += " " + "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + text_list[idx] + "}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def save_visual_results(input, cls_per_token_score, method_name, expl_path='expl', suffix=''):
    input_mask = input['input_mask']
    expl_dir = '/media/data2/hila_chefer/mmf/experiments/'+expl_path

    bbox_scores = cls_per_token_score[0, input_mask.sum(1):]
    _, top_5_bboxes_indices = bbox_scores.topk(k=5, dim=-1)
    image_file_path = '/media/data2/hila_chefer/env_MMF/datasets/coco/subset_val/images/val2014/' + \
                      input['image_info_0']['feature_path'][0] + '.jpg'
    img = cv2.imread(image_file_path)
    for index in top_5_bboxes_indices:
        [x, y, w, h] = input['image_info_0']['bbox'][0][index]
        # cv2.rectangle(img, (int(x), int(y)), ((int(x + w), int(y + h))), (255, 0, 0), 5)
        cv2.rectangle(img, (int(x), int(y)), ((int(w), int(h))), (255, 0, 0), 5)
    cv2.imwrite(
        expl_dir + input['image_info_0']['feature_path'][0] + '_max_{0}_{1}.jpg'.format(method_name, suffix), img)

    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    bbox_scores = (bbox_scores - bbox_scores.min()) / (bbox_scores.max() - bbox_scores.min())
    for index in range(len(bbox_scores)):
        [x, y, w, h] = input['image_info_0']['bbox'][0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor) * bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(curr_score_tensor, new_score_tensor)
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    cv2.imwrite(
        expl_dir + input['image_info_0']['feature_path'][0] + '_{0}.jpg'.format(method_name), img)


    token_scores = cls_per_token_score[0, : input_mask.sum(1)]
    generate(input['tokens'][0], token_scores, expl_dir + input['image_info_0']['feature_path'][0] + '_{0}_{1}.tex'.format(method_name, suffix))


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
