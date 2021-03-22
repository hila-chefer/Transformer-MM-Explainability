import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from DETR.datasets.coco import *
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import os
import random
import cv2

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

from scipy import ndimage as ndi

def gaussian_threshold(image, block_size=11, offset=-0.6, cval=0.1):
    # automatically determine sigma which covers > 99% of distribution
    thresh_image = np.zeros(image.shape, 'double')
    sigma = (block_size - 1) / 6.0
    ndi.gaussian_filter(image, sigma, output=thresh_image, mode='reflect', cval=cval)
    return thresh_image - offset



def evaluate(model, gen, im, device, image_id = None):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    if keep.nonzero().shape[0] <= 1:
        return

    outputs['pred_boxes'] = outputs['pred_boxes'].cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #     lambda self, input, output: enc_attn_weights.append(output[1])
        # ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    for layer in model.transformer.encoder.layers:
        hook = layer.self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        )
        hooks.append(hook)

    model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[-1]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    img_np = np.array(im).astype(np.float)


    ########## ours #############

    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_ours(img, idx, use_lrp=False)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_transformer_att.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        print(idx, CLASSES[probas[idx].argmax()])
    for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
        fig, ax = plt.subplots(ncols=1, nrows=1)
        alpha = 0.5
        img_out = (255 - img_np) * alpha + img_np
        img_out[int(ymin.item()):int(ymax.item()), int(xmin.item()):int(xmax.item())] = img_np[int(ymin.item()):int(
            ymax.item()), int(xmin.item()):int(xmax.item())]
        img_out = img_out.clip(max=255, min=0).astype(np.uint8)
        ax.imshow(img_out, aspect='auto')
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='black', linewidth=3))
        ax.axis('off')

        id_str = '' if image_id == None else image_id
        fig.tight_layout()
        plt.savefig('decoder_visualization/{0}_orig_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_ours(img, idx, use_lrp=False)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_ours_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## ours #############

    ########## ours- with Otsu thresholding #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_ours(img, idx, use_lrp=False)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    #     Res_img = cam.reshape(h, w)
    #     Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
    #     ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     cam = torch.from_numpy(th).to(outputs["pred_logits"].device).type(torch.float32)
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_ours_otsu.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_ours(img, idx, use_lrp=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        Res_img = cam.reshape(h, w)
        Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
        ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cam = torch.from_numpy(th).to(outputs["pred_logits"].device).type(torch.float32)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_ours_otsu_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## ours- with Otsu thresholding #############

    ######### transformer_att #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_transformer_att(img, idx)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_transformer_att.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_transformer_att(img, idx)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_transformer_att_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## transformer_att #############

    ########## rollout #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_rollout(img, idx)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_rollout.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_rollout(img, idx)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_rollout_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## rollout #############

    ########## raw attn #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_raw_attn(img, idx)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_raw_attn.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_raw_attn(img, idx)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_raw_attn_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## raw attn #############

    ########## partial_lrp #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_partial_lrp(img, idx)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_partial_lrp.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_partial_lrp(img, idx)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_partial_lrp_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ########## partial_lrp #############

    ########## attn_gradca #############
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_attn_gradcam(img, idx)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     ax.imshow(cam.view(h, w).data.cpu().numpy())
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # id_str = '' if image_id == None else image_id
    # plt.savefig('decoder_visualization/{0}_attn_gradca.png'.format(id_str))
    # fig.tight_layout()

    for idx in keep.nonzero():
        fig, ax = plt.subplots(ncols=1, nrows=1)
        cam = gen.generate_attn_gradcam(img, idx)
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap, aspect='auto')
        ax.axis('off')
        id_str = '' if image_id == None else image_id
        plt.savefig('decoder_visualization/{0}_gradcam_{1}.png'.format(id_str, idx.item()), bbox_inches='tight', pad_inches=0)
        fig.tight_layout()
        plt.close(fig)
    ######### attn_gradca #############

def build_dataset(image_set):
    root = Path('/media/data2/hila_chefer/detr/coco/')
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=None)
    return dataset

# @torch.no_grad()
def evaluate_val_set(model, device):
    model.eval()
    directory = r'/media/data2/hila_chefer/DETR/coco/val2017/'
    gen = Generator(model)
    ids_list = os.listdir(directory)
    random_index = list(range(len(ids_list)))
    random.shuffle(random_index)
    # for index,filename in enumerate(reversed()):
    for i in random_index:
        filename = ids_list[i]
        im = Image.open(directory + filename)
        evaluate(model, gen, im, device, filename)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
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

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


if __name__ == '__main__':
    import argparse
    import DETR.util.misc as utils
    from DETR.models import build_model
    from DETR.modules.ExplanationGenerator import Generator

    # model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    # loading DETR line 0 from README model zoo- only detection
    # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    model_without_ddp.load_state_dict(checkpoint['model'])


    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = 'http://images.cocodataset.org/val2017/000000216516.jpg'
    # url = "http://images.cocodataset.org/val2017/000000281759.jpg"
    # url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    # url = "http://images.cocodataset.org/val2017/000000037777.jpg"
    # url = 'http://images.cocodataset.org/val2017/000000448448.jpg'
    # url = 'http://images.cocodataset.org/val2017/000000144114.jpg'
    # url = 'http://images.cocodataset.org/val2017/000000306700.jpg'
    # url = 'http://images.cocodataset.org/val2017/000000467511.jpg'
    # url = 'http://images.cocodataset.org/val2017/000000359937.jpg'
    url = 'http://images.cocodataset.org/val2017/000000192191.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    # specific image by id
    gen = Generator(model)
    evaluate(model_without_ddp, gen, im, device=device, image_id="oven")

    # all coco val images
    # evaluate_val_set(model_without_ddp, device="cuda")

    # from pycocotools.coco import COCO
    # import numpy as np
    # import skimage.io as io
    # import random
    # import os
    # import cv2
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    #
    # ### For visualizing the outputs ###
    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec
    #
    # dataDir = '/media/data2/hila_chefer/DETR/coco'
    # dataType = 'val2017'
    # annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    #
    # # Initialize the COCO api for instance annotations
    # coco = COCO(annFile)
    #
    # # Load the categories in a variable
    # catIDs = coco.getCatIds()
    # cats = coco.loadCats(catIDs)
    #
    # print(cats)
    #
    # id = 39769
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # I = Image.open(requests.get(url, stream=True).raw)
    #
    #
    # def getClassName(classID, cats):
    #     for i in range(len(cats)):
    #         if cats[i]['id'] == classID:
    #             return cats[i]['name']
    #     return "None"
    # # Load and display instance annotations
    # annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
    # anns = coco.loadAnns(annIds)
    # filterClasses = ['cat']
    # mask = np.zeros((I.height, I.width))
    # for i in range(len(anns)):
    #     className = getClassName(anns[i]['category_id'], cats)
    #     try:
    #         pixel_value = filterClasses.index(className) + 1
    #     except:
    #         print("Skipping {}".format(className))
    #         continue
    #     mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
    #     print(mask.shape)
    # plt.imshow(mask)
    # plt.show()
    #
    # img = cv2.cvtColor(np.array(I), cv2.COLOR_RGB2BGR)
    # print(img.shape)
    # for i in range(len(anns)):
    #     className = getClassName(anns[i]['category_id'], cats)
    #     if className in filterClasses:
    #         [x, y, w, h] = anns[i]['bbox']
    #         cv2.rectangle(img, (int(x), int(y)), ((int(x + w), int(y + h))), (255, 0, 0), 5)

