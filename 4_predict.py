import sys,os
import torch
import numpy as np
from PIL import Image
import json

from models import build_DABDETR, build_dab_deformable_detr
from util.slconfig import SLConfig
from datasets import build_dataset
from util import box_ops
import datasets.transforms as T

model_config_path = sys.argv[1]
model_checkpoint_path = sys.argv[2]
val_dataset_path = sys.argv[3]
val_annotations = sys.argv[4]
pred_json = sys.argv[5]

# See our Model Zoo section in README.md for more details about our pretrained mod

args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_dab_deformable_detr(args)
device = torch.device('cuda')
model = model.to(device)
checkpoint = torch.load(model_checkpoint_path, map_location='cuda')
model.load_state_dict(checkpoint['model'])

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open(val_annotations) as f:
    data = json.load(f)

output_dict = dict()

for ele in data['images']:
    source_dir = os.path.join(val_dataset_path,ele['file_name'])
    image =  Image.open(source_dir).convert('RGB')
    
    w,h = image.size
    image, _ = transform(image, None)
    output = model(image[None].to(device))
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).to(device))[0]

    scores = output['scores']
    labels = output['labels']
    boxes = output['boxes']

    for s_ele in boxes:
        s_ele[0]*=w
        s_ele[1]*=h
        s_ele[2]*=w
        s_ele[3]*=h

    pred_dict = {
    'boxes': boxes.tolist(),
    'labels': labels.tolist(),
    'scores': scores.tolist() 
    }
    output_dict[ele['file_name']] = pred_dict

with open(pred_json,'w') as fp:
    json.dump(output_dict,fp,indent=4)