import torch
from DETR.modules.ExplanationGenerator import Generator, GeneratorAlbationNoAgg
import numpy as np
from PIL import Image

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

class MaskGenerator:
    def __init__(self, model):
        self.gen = Generator(model)
        self.abl = GeneratorAlbationNoAgg(model)
        self.model = model

    def get_panoptic(self, samples, targets, method):
        # propagate through the model
        outputs = self.model(samples)

        # keep only predictions with 0.8+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5

        ########### for visualizations
        boxes = outputs['pred_boxes'].cpu()
        im = samples.tensors[0].permute(1, 2, 0).data.cpu().numpy()
        im = (im - im.min()) / (im.max() - im.min())
        im = np.uint8(im * 255)
        im = Image.fromarray(im)
        # im = T.ToPILImage()(samples.tensors[0])
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(boxes[0, keep], im.size)
        ############ for visualizations


        if keep.nonzero().shape[0] <= 1:
            print("no segmentation")

        # use lists to store the outputs via up-values
        vis_shape, target_shape = [], []

        hooks = [
            self.model.transformer.register_forward_hook(
                lambda self, input, output: vis_shape.append(output[1])
            ),
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: target_shape.append(output)
            )
        ]

        self.model(samples)

        for hook in hooks:
            hook.remove()

        h, w = vis_shape[0].shape[-2:]
        # print("h,w", h, w)
        target_shape = target_shape[0]
        import cv2
        masks = torch.ones(1, 100, h, w).to(outputs["pred_logits"].device) * (-1)
        for idx in keep.nonzero():
            if method == 'ours_with_lrp':
                cam = self.gen.generate_ours(samples, idx, use_lrp=True)
            elif method == 'ours_no_lrp':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False)
            elif method == 'ablation_no_self_in_10':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False, apply_self_in_rule_10=False)
            elif method == 'ablation_no_aggregation':
                cam = self.abl.generate_ours_abl(samples, idx, use_lrp=False, normalize_self_attention=False)
            elif method == 'ours_no_lrp_no_norm':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False, normalize_self_attention=False)
            elif method == 'transformer_att':
                cam = self.gen.generate_transformer_att(samples, idx)
            elif method == 'raw_attn':
                cam = self.gen.generate_raw_attn(samples, idx)
            elif method == 'attn_gradcam':
                cam = self.gen.generate_attn_gradcam(samples, idx)
            elif method == 'rollout':
                cam = self.gen.generate_rollout(samples, idx)
            elif method == 'partial_lrp':
                cam = self.gen.generate_partial_lrp(samples, idx)
            else:
                print("please provide a valid explainability method")
                return

            # Otsu
            cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
            Res_img = cam.reshape(h, w)
            Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
            ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cam = torch.from_numpy(th).to(outputs["pred_logits"].device).type(torch.float32)
            masks[0, idx] = cam

        # import matplotlib.pyplot as plt
        #
        # plt.clf()
        # print("Bboxes scaled:::::", len(bboxes_scaled))
        # if len(bboxes_scaled) > 1:
        #     fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        #     for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        #         ax = ax_i[0]
        #         cam = self.gen.generate_ours(samples, idx, use_lrp=False).reshape(h,w)
        #         cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        #         Res_img = cam.reshape(h, w)
        #         Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
        #         import cv2
        #         ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #         cam = torch.from_numpy(Res_img)
        #         cam[cam < ret] = 0
        #         cam[cam >= ret] = 1
        #         ax.imshow(cam)
        #         ax.axis('off')
        #         ax.set_title(f'query id: {idx.item()}')
        #         ax = ax_i[1]
        #         ax.imshow(im)
        #         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                    fill=False, color='blue', linewidth=3))
        #         ax.axis('off')
        #         ax.set_title(CLASSES[probas[idx].argmax()])
        #
        #     plt.show()
        #     plt.savefig('decoder_visualization/_ours_seg.png')

        outputs['pred_masks'] = masks

        return outputs


