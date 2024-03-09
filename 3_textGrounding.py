import argparse
import json
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os, subprocess 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import sys
import pytorch_fid

device = "cuda"

with open('../hw1_dataset/annotations/train.json') as f:
    annotations = json.load(f)
orig_json_img_list = annotations['images']
orig_json_bbox_list = annotations['annotations']

with open('../result_1/ImageCaption_DataAugment.json') as f:
    ImgCaption_DA = json.load(f)

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale

def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas

def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] , strict=False)
    autoencoder.load_state_dict( saved_ckpt["autoencoder"], strict=False )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"], strict=False  )
    diffusion.load_state_dict( saved_ckpt["diffusion"], strict=False  )

    return model, autoencoder, text_encoder, diffusion, config

def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask

def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image

@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)

    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 

@torch.no_grad()
def run(meta, config, save_name, starting_noise=None):

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    batch = prepare_batch(meta, config.batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )
    if args.negative_prompt is not None:
        uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )
    
    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).cuda()
        
        input_image = F.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
    
    # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,

            )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save - - - - - #
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    print(image_ids)
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = save_name  +'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(os.path.join(output_folder, img_name))

def Num_Cat_NeedHave():
    UniqueNum_Img_byCat = dict()
    GenNum_Img_byCat = dict()
    Cat_list = annotations['categories']
    for cat in Cat_list[1:]:
        UniqueNum_Img_byCat[cat['id']] = 0
        GenNum_Img_byCat[cat['id']] = 0
    for img in annotations['images']:
        img_id = img['id']
        cat_set = set([bbox['category_id'] for bbox in annotations['annotations'] if bbox['image_id'] == img_id])
        for cat_id in cat_set:
            UniqueNum_Img_byCat[cat_id]+=1
    NeedNum_Img_byCat = UniqueNum_Img_byCat
    max_num = max(UniqueNum_Img_byCat.values())
    for cat in annotations['categories'][1:]:
        NeedNum_Img_byCat[cat['id']] = round(max_num - NeedNum_Img_byCat[cat['id']])

    for prompt in ImgCaption_DA:
        id_ = [x for x in range(len(Cat_list)) if Cat_list[x]['name'] == prompt['label']][0]
        GenNum_Img_byCat[id_] += 1
    return NeedNum_Img_byCat, GenNum_Img_byCat

def Meta_Text_Grounding(Gen_ele, prompt_type, imgid, option):
    rand_num = random.randint(1,6)
    bboxes = []
    if option == 'r':
        save_folder_name = 'randomboxes/generation_box_text/'
        for x in range(rand_num):
            [x1, x2] = sorted([random.random(), random.random()])
            [y1, y2] = sorted([random.random(), random.random()])
            bboxes.append([x1,y1,x2,y2])
    else:
        save_folder_name = 'originalboxes/generation_box_text/'
        bboxes = Gen_ele['bboxes']
    dict_ele = dict(
        ckpt = '../checkpoint_generation_text.pth',
        prompt = Gen_ele[prompt_type],
        phrases = Gen_ele['label'],
        locations = bboxes,
        alpha_type = [0.3, 0.0, 0.7],
        save_folder_name = save_folder_name,
        file_name = Gen_ele['image'])
    return dict_ele

def Meta_Image_Grounding(Gen_ele, prompt_type, imgid, option):
    rand_num = random.randint(1,6)
    bboxes = []
    if option == 'r':
        save_folder_name = 'randomboxes/generation_box_image/'
        for x in range(rand_num):
            [x1, x2] = sorted([random.random(), random.random()])
            [y1, y2] = sorted([random.random(), random.random()])
            bboxes.append([x1,y1,x2,y2])
    else:
        save_folder_name = 'originalboxes/generation_box_image/'
        bboxes = Gen_ele['bboxes']
    for x in range(rand_num):
        [x1, x2] = sorted([random.random(), random.random()])
        [y1, y2] = sorted([random.random(), random.random()])
        bboxes.append([x1,y1,x2,y2])
    dict_ele = dict(
        ckpt = '../checkpoint_generation_text_image.pth',
        prompt = Gen_ele[prompt_type],
        images = [os.path.join('../hw1_dataset/train', Gen_ele['image'])],
        phrases = Gen_ele['label'],
        locations = bboxes,
        alpha_type = [1.0, 0.0, 0.0],
        save_folder_name = save_folder_name,
        file_name = Gen_ele['image'])
    return dict_ele

if __name__ == '__main__':
    
    # Reference the inference code 
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="../result_4", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=5, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    args = parser.parse_args()

    
    # open the result of the generating texts from the ImgCaption.py
    starting_noise = torch.randn(args.batch_size,4,64,64).to(device)
    starting_noise = None
    
    # get optimal prompt
    opt_prompt_type=os.listdir('../result_23/generation_box_image')[0]

    # Next, Doing Data Augmentation 
    output_train_gen_json = {"categories":None, "images":[], "annotations":[]}
    output_train_gen_json['categories'] = annotations['categories']

    # First, Calculate the Avg Number 
    Orig_CatNum_List = [{'id':cat['id'],'category':cat['name'],'number':0} for cat in annotations['categories'][1:]] 
    
    # Get the categorical number of the original dataset 
    NeedNum_Img_byCat, GenNum_Img_byCat  = Num_Cat_NeedHave()
    run_round_by_Cat = {x:round(NeedNum_Img_byCat[x]/GenNum_Img_byCat[x]) for x in NeedNum_Img_byCat.keys()}

    # Generate Images by Text Grounding, for Data Augmentation 
    id_img, id_bbox1, id_bbox2, id_bbox3, id_bbox4 = len(orig_json_img_list), len(orig_json_bbox_list), len(orig_json_bbox_list), len(orig_json_bbox_list), len(orig_json_bbox_list)

    # for Text Grounding(original boxes)
    with open('../hw1_dataset/annotations/train.json') as f:
        output_text_json_o = json.load(f)
    
    # for Image Grounding(original boxes)
    with open('../hw1_dataset/annotations/train.json') as f:
        output_img_json_o = json.load(f) 

    # for Text Grounding(random boxes)
    with open('../hw1_dataset/annotations/train.json') as f:
        output_text_json_r = json.load(f)
    
    # for Image Grounding(random boxes)
    with open('../hw1_dataset/annotations/train.json') as f:
        output_img_json_r = json.load(f) 

    for cat_id in run_round_by_Cat.keys():
        for round in range(run_round_by_Cat[cat_id]):
            for caption in ImgCaption_DA:
                if caption['label']== annotations['categories'][cat_id]['name']:
                    # Generate Image and corresponding annotations
                    dict_ele_text_grounding_r = Meta_Text_Grounding(caption, opt_prompt_type, id_img, 'r')
                    dict_ele_image_grounding_r = Meta_Image_Grounding(caption, opt_prompt_type, id_img, 'r')
                    dict_ele_text_grounding_o = Meta_Text_Grounding(caption, opt_prompt_type, id_img, 'o')
                    dict_ele_image_grounding_o = Meta_Image_Grounding(caption, opt_prompt_type, id_img, 'o')                   
                    run(dict_ele_text_grounding_r, args, dict_ele_text_grounding_r['file_name'][:-4]+str(id_img), starting_noise)
                    run(dict_ele_image_grounding_r, args, dict_ele_image_grounding_r['file_name'][:-4]+str(id_img), starting_noise)
                    run(dict_ele_text_grounding_o, args, dict_ele_text_grounding_o['file_name'][:-4]+str(id_img), starting_noise)
                    run(dict_ele_image_grounding_o, args, dict_ele_image_grounding_o['file_name'][:-4]+str(id_img), starting_noise)
                    output_text_json_r['images'].append({"id":id_img,"license":1,"file_name":dict_ele_text_grounding_r['file_name'][:-4]+str(id_img)+'.png', "height":512, "width":512, "date_captured":"2020-11-18T19:53:47+00:00" })
                    output_img_json_r['images'].append({"id":id_img,"license":1,"file_name":dict_ele_image_grounding_r['file_name'][:-4]+str(id_img)+'.png', "height":512, "width":512, "date_captured":"2020-11-18T19:53:47+00:00" })
                    output_text_json_o['images'].append({"id":id_img,"license":1,"file_name":dict_ele_text_grounding_o['file_name'][:-4]+str(id_img)+'.png', "height":512, "width":512, "date_captured":"2020-11-18T19:53:47+00:00" })
                    output_img_json_o['images'].append({"id":id_img,"license":1,"file_name":dict_ele_image_grounding_o['file_name'][:-4]+str(id_img)+'.png', "height":512, "width":512, "date_captured":"2020-11-18T19:53:47+00:00" })
                    for location in dict_ele_text_grounding_r["locations"]:
                        bbox = [location[0], location[1], location[2]-location[0], location[3]-location[1]]
                        bbox = [int(ele*512) for ele in bbox]
                        output_text_json_r['annotations'].append({"id":id_bbox1,"image_id":id_img,"category_id":cat_id, "bbox":bbox, "area":bbox[2]*bbox[3], "segmentation":[],"iscrowd":0})
                        id_bbox1+=1
                    for location in dict_ele_image_grounding_r["locations"]:
                        bbox = [location[0], location[1], location[2]-location[0], location[3]-location[1]]
                        bbox = [int(ele*512) for ele in bbox]
                        output_img_json_r['annotations'].append({"id":id_bbox2, "image_id":id_img,"category_id":cat_id, "bbox":bbox, "area":bbox[2]*bbox[3], "segmentation":[],"iscrowd":0})
                        id_bbox2+=1
                    for location in dict_ele_text_grounding_o["locations"]:
                        bbox = [location[0], location[1], location[2]-location[0], location[3]-location[1]]
                        bbox = [int(ele*512) for ele in bbox]
                        output_text_json_o['annotations'].append({"id":id_bbox3,"image_id":id_img,"category_id":cat_id, "bbox":bbox, "area":bbox[2]*bbox[3], "segmentation":[],"iscrowd":0})
                        id_bbox3+=1
                    for location in dict_ele_image_grounding_o["locations"]:
                        bbox = [location[0], location[1], location[2]-location[0], location[3]-location[1]]
                        bbox = [int(ele*512) for ele in bbox]
                        output_img_json_o['annotations'].append({"id":id_bbox4, "image_id":id_img,"category_id":cat_id, "bbox":bbox, "area":bbox[2]*bbox[3], "segmentation":[],"iscrowd":0})
                        id_bbox4+=1
                    id_img+=1
    with open('../result_4/train_Text_Grounding(random).json','w') as f:
        json.dump(output_text_json_r, f, indent=2 )
    with open('../result_4/train_Image_Grounding(random).json','w') as f:
        json.dump(output_img_json_r, f, indent=2 )
    with open('../result_4/train_Text_Grounding(original).json','w') as f:
        json.dump(output_text_json_o, f, indent=2 )
    with open('../result_4/train_Image_Grounding(original).json','w') as f:
        json.dump(output_img_json_o, f, indent=2 )

                
