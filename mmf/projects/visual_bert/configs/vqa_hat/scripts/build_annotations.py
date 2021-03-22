import numpy as np
from mmf.utils.file_io import PathManager
import os
from glob import glob
import json
from tqdm import tqdm


def _load_jsonl(path):
    with PathManager.open(path, "r") as f:
        db = f.readlines()
        for idx, line in enumerate(db):
            db[idx] = json.loads(line.strip("\n"))
    return db


vqahat_path = "/media/data2/hila_chefer/env_MMF/datasets/vqa_hat"
vqa2_path = "/media/data2/hila_chefer/env_MMF/datasets/vqa2"

print("Loading VQA 2.0 data...")
with PathManager.open(os.path.join(vqa2_path, "defaults/annotations/imdb_train2014.npy"), "rb") as f:
    train_vqa_db = np.load(f, allow_pickle=True)

with PathManager.open(os.path.join(vqa2_path, "defaults/annotations/imdb_val2014.npy"), "rb") as f:
    val_vqa_db = np.load(f, allow_pickle=True)

print("Loading VQA-HAT data...")
train_vqahat_imgs = glob(os.path.join(vqahat_path, "defaults/images/vqahat_train/*.png"))
val_vqahat_imgs = glob(os.path.join(vqahat_path, "defaults/images/vqahat_val/*.png"))

print("Loading VQA 1.0 data...")
train_vqahat_ann = _load_jsonl(os.path.join(vqahat_path, "defaults/annotations/mscoco_train2014_annotations.json"))[0]['annotations']
val_vqahat_ann = _load_jsonl(os.path.join(vqahat_path, "defaults/annotations/mscoco_val2014_annotations.json"))[0]['annotations']
train_vqahat_qst = _load_jsonl(os.path.join(vqahat_path, "defaults/annotations/MultipleChoice_mscoco_train2014_questions.json"))[0]['questions']
val_vqahat_qst = _load_jsonl(os.path.join(vqahat_path, "defaults/annotations/MultipleChoice_mscoco_val2014_questions.json"))[0]['questions']

print("Indexing misc...")
train_vqa_qid2idx = {}
train_vqa_iid2idx = {}
val_vqa_qid2idx = {}
val_vqa_iid2idx = {}
for i in range(1, len(train_vqa_db)):
    iid = train_vqa_db[i]['image_id']
    qid = train_vqa_db[i]["question_id"]
    train_vqa_qid2idx[qid] = i
    train_vqa_iid2idx[iid] = i
for i in range(1, len(val_vqa_db)):
    iid = val_vqa_db[i]['image_id']
    qid = val_vqa_db[i]["question_id"]
    val_vqa_qid2idx[qid] = i
    val_vqa_iid2idx[iid] = i

train_vqahat_qid2idx = {}
val_vqahat_qid2idx = {}
for i in range(len(train_vqahat_ann)):
    qid = train_vqahat_ann[i]["question_id"]
    train_vqahat_qid2idx[qid] = i
for i in range(len(val_vqahat_ann)):
    qid = val_vqahat_ann[i]["question_id"]
    val_vqahat_qid2idx[qid] = i

print("Preparing training annotations...")
train_vqahat_annotations = []
for img_name in tqdm(train_vqahat_imgs):
    qid = int(img_name.split('/')[-1].split('.')[0].split('_')[0])
    feature_path = train_vqa_db[train_vqa_iid2idx[train_vqahat_ann[train_vqahat_qid2idx[qid]]["image_id"]]]['feature_path']

    sample = {}
    sample['image_name'] = train_vqa_db[train_vqa_iid2idx[train_vqahat_ann[train_vqahat_qid2idx[qid]]["image_id"]]]['image_name']
    sample['image_id'] = train_vqa_db[train_vqa_iid2idx[train_vqahat_ann[train_vqahat_qid2idx[qid]]["image_id"]]]['image_id']
    sample['question_id'] = train_vqahat_ann[train_vqahat_qid2idx[qid]]['question_id']
    sample['feature_path'] = train_vqa_db[train_vqa_iid2idx[train_vqahat_ann[train_vqahat_qid2idx[qid]]["image_id"]]]['feature_path']
    question = train_vqahat_qst[train_vqahat_qid2idx[qid]]['question']
    question = question if question[-1] == '?' else question + '?'
    sample['question_str'] = question
    sample['question_tokens'] = sample['question_str'].lower().split(' ')
    sample['all_answers'] = [x['answer'] for x in train_vqahat_ann[train_vqahat_qid2idx[qid]]['answers']]
    sample['ocr_tokens'] = train_vqa_db[train_vqa_iid2idx[train_vqahat_ann[train_vqahat_qid2idx[qid]]["image_id"]]]['ocr_tokens']
    sample['answers'] = sample['all_answers']

    train_vqahat_annotations.append(sample)

train_save_path = os.path.join(vqahat_path, 'defaults/annotations/train')
print(f'Saving training annotations to - "{train_save_path}.npy"')
np.save(train_save_path, train_vqahat_annotations, allow_pickle=True)

print("Preparing validation annotations...")
val_vqahat_annotations = []
for img_name in tqdm(val_vqahat_imgs):
    qid = int(img_name.split('/')[-1].split('.')[0].split('_')[0])
    feature_path = val_vqa_db[val_vqa_iid2idx[val_vqahat_ann[val_vqahat_qid2idx[qid]]["image_id"]]]['feature_path']

    sample = {}
    sample['image_name'] = val_vqa_db[val_vqa_iid2idx[val_vqahat_ann[val_vqahat_qid2idx[qid]]["image_id"]]]['image_name']
    sample['image_id'] = val_vqa_db[val_vqa_iid2idx[val_vqahat_ann[val_vqahat_qid2idx[qid]]["image_id"]]]['image_id']
    sample['question_id'] = val_vqahat_ann[val_vqahat_qid2idx[qid]]['question_id']
    sample['feature_path'] = val_vqa_db[val_vqa_iid2idx[val_vqahat_ann[val_vqahat_qid2idx[qid]]["image_id"]]]['feature_path']
    question = val_vqahat_qst[val_vqahat_qid2idx[qid]]['question']
    question = question if question[-1] == '?' else question + '?'
    sample['question_str'] = question
    sample['question_tokens'] = sample['question_str'].lower().split(' ')
    sample['all_answers'] = [x['answer'] for x in val_vqahat_ann[val_vqahat_qid2idx[qid]]['answers']]
    sample['ocr_tokens'] = val_vqa_db[val_vqa_iid2idx[val_vqahat_ann[val_vqahat_qid2idx[qid]]["image_id"]]]['ocr_tokens']
    sample['answers'] = sample['all_answers']

    val_vqahat_annotations.append(sample)

val_save_path = os.path.join(vqahat_path, 'defaults/annotations/val')
print(f'Saving validation annotations to - "{val_save_path}.npy"')
np.save(val_save_path, val_vqahat_annotations, allow_pickle=True)

print("Done.")
