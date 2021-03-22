from lxmert.lxmert.src.tasks import vqa_data
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import GeneratorOurs, GeneratorBaselines, GeneratorOursAblationNoAggregation
import random
from lxmert.lxmert.src.param import args

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

class ModelPert:
    def __init__(self, COCO_val_path, use_lrp=False):
        self.COCO_VAL_PATH = COCO_val_path
        self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        if use_lrp:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        self.vqa_dataset = vqa_data.VQADataset(splits="valid")

        self.pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.pert_acc = [0] * len(self.pert_steps)

    def forward(self, item):
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        self.image_file_path = image_file_path
        self.image_id = item['img_id']
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        self.image_boxes_len = features.shape[1]
        self.bboxes = output_dict.get("boxes")
        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            visual_feats=features.to("cuda"),
            visual_pos=normalized_boxes.to("cuda"),
            token_type_ids=inputs.token_type_ids.to("cuda"),
            return_dict=True,
            output_attentions=False,
        )
        return self.output

    def perturbation_image(self, item, cam_image, cam_text, is_positive_pert=False):
        if is_positive_pert:
            cam_image = cam_image * (-1)
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        for step_idx, step in enumerate(self.pert_steps):
            # find top step boxes
            curr_num_boxes = int((1 - step) * self.image_boxes_len)
            _, top_bboxes_indices = cam_image.topk(k=curr_num_boxes, dim=-1)
            top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()

            curr_features = features[:, top_bboxes_indices, :]
            curr_pos = normalized_boxes[:, top_bboxes_indices, :]

            output = self.lxmert_vqa(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
                visual_feats=curr_features.to("cuda"),
                visual_pos=curr_pos.to("cuda"),
                token_type_ids=inputs.token_type_ids.to("cuda"),
                return_dict=True,
                output_attentions=False,
            )

            answer = self.vqa_answers[output.question_answering_score.argmax()]
            accuracy = item["label"].get(answer, 0)
            self.pert_acc[step_idx] += accuracy

        return self.pert_acc

    def perturbation_text(self, item, cam_image, cam_text, is_positive_pert=False):
        if is_positive_pert:
            cam_text = cam_text * (-1)
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        for step_idx, step in enumerate(self.pert_steps):
            # we must keep the [CLS] token in order to have the classification
            # we also keep the [SEP] token
            cam_pure_text = cam_text[1:-1]
            text_len = cam_pure_text.shape[0]
            # find top step tokens, without the [CLS] token and the [SEP] token
            curr_num_tokens = int((1 - step) * text_len)
            _, top_bboxes_indices = cam_pure_text.topk(k=curr_num_tokens, dim=-1)
            top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()

            # add back [CLS], [SEP] tokens
            top_bboxes_indices = [0, cam_text.shape[0] - 1] +\
                                 [top_bboxes_indices[i] + 1 for i in range(len(top_bboxes_indices))]
            # text tokens must be sorted for positional embedding to work
            top_bboxes_indices = sorted(top_bboxes_indices)

            curr_input_ids = inputs.input_ids[:, top_bboxes_indices]
            curr_attention_mask = inputs.attention_mask[:, top_bboxes_indices]
            curr_token_ids = inputs.token_type_ids[:, top_bboxes_indices]

            output = self.lxmert_vqa(
                input_ids=curr_input_ids.to("cuda"),
                attention_mask=curr_attention_mask.to("cuda"),
                visual_feats=features.to("cuda"),
                visual_pos=normalized_boxes.to("cuda"),
                token_type_ids=curr_token_ids.to("cuda"),
                return_dict=True,
                output_attentions=False,
            )

            answer = self.vqa_answers[output.question_answering_score.argmax()]
            accuracy = item["label"].get(answer, 0)
            self.pert_acc[step_idx] += accuracy

        return self.pert_acc

def main(args):
    model_pert = ModelPert(args.COCO_path, use_lrp=True)
    ours = GeneratorOurs(model_pert)
    baselines = GeneratorBaselines(model_pert)
    oursNoAggAblation = GeneratorOursAblationNoAggregation(model_pert)
    vqa_dataset = vqa_data.VQADataset(splits="valid")
    vqa_answers = utils.get_data(VQA_URL)
    method_name = args.method

    items = vqa_dataset.data
    random.seed(1234)
    r = list(range(len(items)))
    random.shuffle(r)
    pert_samples_indices = r[:args.num_samples]
    iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    test_type = "positive" if args.is_positive_pert else "negative"
    modality = "text" if args.is_text_pert else "image"
    print("runnig {0} pert test for {1} modality with method {2}".format(test_type, modality, args.method))

    for index, item in enumerate(iterator):
        if method_name == 'transformer_att':
            R_t_t, R_t_i = baselines.generate_transformer_attr(item)
        elif method_name == 'attn_gradcam':
            R_t_t, R_t_i = baselines.generate_attn_gradcam(item)
        elif method_name == 'partial_lrp':
            R_t_t, R_t_i = baselines.generate_partial_lrp(item)
        elif method_name == 'raw_attn':
            R_t_t, R_t_i = baselines.generate_raw_attn(item)
        elif method_name == 'rollout':
            R_t_t, R_t_i = baselines.generate_rollout(item)
        elif method_name == "ours_with_lrp_no_normalization":
            R_t_t, R_t_i = ours.generate_ours(item, normalize_self_attention=False)
        elif method_name == "ours_no_lrp":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False)
        elif method_name == "ours_no_lrp_no_norm":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False, normalize_self_attention=False)
        elif method_name == "ours_with_lrp":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=True)
        elif method_name == "ablation_no_self_in_10":
            R_t_t, R_t_i = ours.generate_ours(item, use_lrp=False, apply_self_in_rule_10=False)
        elif method_name == "ablation_no_aggregation":
            R_t_t, R_t_i = oursNoAggAblation.generate_ours_no_agg(item, use_lrp=False, normalize_self_attention=False)
        else:
            print("Please enter a valid method name")
            return
        cam_image = R_t_i[0]
        cam_text = R_t_t[0]
        cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
        cam_text = (cam_text - cam_text.min()) / (cam_text.max() - cam_text.min())
        if args.is_text_pert:
            curr_pert_result = model_pert.perturbation_text(item, cam_image, cam_text, args.is_positive_pert)
        else:
            curr_pert_result = model_pert.perturbation_image(item, cam_image, cam_text, args.is_positive_pert)
        curr_pert_result = [round(res / (index+1) * 100, 2) for res in curr_pert_result]
        iterator.set_description("Acc: {}".format(curr_pert_result))

if __name__ == "__main__":
    main(args)