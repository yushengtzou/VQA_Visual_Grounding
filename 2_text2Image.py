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
import shutil

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
        sample.save(  os.path.join(output_folder, img_name)   )

# Random select the 20 images for assignment 
def extract_twenty_img(json_file):
    with open('../hw1_dataset/annotations/train.json') as f:
        original_file = json.load(f)
        cat_list = [ele['name'] for ele in original_file['categories'][1:]]
    output_list = list()
    for x in cat_list:
        locals()['list_num_'+x] = [idx for idx in range(len(json_file)) if json_file[idx]['label'] == x]
        output_list.append({"category":x, "sel_num":random.sample(locals()['list_num_'+x],20)})
    return output_list

# Output the objects for generating the image, Text Grounding
def Meta_Text_Grounding(json_file, prompt_type, category):
    output_list = []
    for ele in json_file:
        dict_ele = dict(
            ckpt = '../checkpoint_generation_text.pth',
            prompt = ele[prompt_type],
            phrases = ele['label'],
            locations = ele['bboxes'],
            alpha_type = [0.3, 0.0, 0.7],
            save_folder_name = 'generation_box_text/' + prompt_type + '/' + category + '/',
            file_name = ele['image'])
        output_list.append(dict_ele)
    return output_list

def Meta_Image_Grounding(json_file, prompt_type, category):
    output_list = []
    for ele in json_file:
        dict_ele = dict(
            ckpt = "../checkpoint_generation_text_image.pth",
            prompt = ele[prompt_type],
            images = [os.path.join('../hw1_dataset/train', ele['image'])],
            phrases = ele['label'],
            locations = ele['bboxes'],
            alpha_type = [1.0, 0.0, 0.0],
            save_folder_name= 'generation_box_image/' + prompt_type + '/' + category + '/',
            file_name = ele['image'])
        #print(dict_ele['images'])
        output_list.append(dict_ele)
    return output_list

# Resize Image and comput the FID score
def FID(ImageCaptionSelected, prompt_type, generation_dir):
    # Make new dir to store the dataset to compute FID
    resize_orig_path = os.path.join(os.path.join(os.path.join('../result_23', generation_dir), 'original_dataset'), prompt_type)
    resize_gen_path = os.path.join(os.path.join(os.path.join('../result_23',generation_dir), 'generated_imgs'), prompt_type)
    os.makedirs(resize_orig_path, exist_ok=True)
    os.makedirs(resize_gen_path, exist_ok=True)

    for ele in ImageCaptionSelected:
        img_name = ele["image"][:-4]
        # resize the original image
        nmv_orig_img_path = os.path.join('../hw1_dataset/train',img_name+'.jpg')
        mv_orig_img_path = os.path.join(resize_orig_path, img_name+'.jpg')
        img = Image.open(nmv_orig_img_path).resize((512,512))
        img.save(mv_orig_img_path)

        nmv_gen_img_path = os.path.join('../result_23', generation_dir+'/'+ prompt_type+'/'+ ele['label']+ '/' + img_name+'.png')
        mv_gen_img_path = os.path.join(resize_gen_path,img_name+'.png')
        gen_img = Image.open(nmv_gen_img_path).resize((512,512))
        gen_img.save(mv_gen_img_path)
    cmd = 'python -m pytorch_fid' + ' ' + resize_orig_path + ' ' + resize_gen_path
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE, universal_newlines=True).stdout
    index = result.rfind('FID:')
    
    return round(float(result[index+4:]), 10)

if __name__ == '__main__':

    device = "cuda"
    with open('../hw1_dataset/annotations/train.json') as f:
        annotations = json.load(f)
    orig_json_img_list = annotations['images']
    orig_json_bbox_list = annotations['annotations']

    with open('../result_1/ImageCaption_Evaluation.json') as f:
        ImgCaption_Eval = json.load(f)

    # Reference the inference code 
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="../result_23", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=5, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    args = parser.parse_args()

    
    # open the result of the generating texts from the ImgCaption.py
    starting_noise = torch.randn(args.batch_size,4,64,64).to(device)
    starting_noise = None
    
    
    os.makedirs('../result_23', exist_ok=True)
    
    # First, randomly select the twenty images 
    selected_num_list = extract_twenty_img(ImgCaption_Eval) # about the number of selected img of each cat
    CombineImgCaptionSelected = [] # combine the selected_num_list 
    
    # Random Selected the number of image of each group, and generate images 
    for dict_ele in selected_num_list:
        print("Now is generating the image of the category(Text Grounding): "+ dict_ele['category'])
        sel_num = dict_ele['sel_num']
        ImgCaptionSelected = [ImgCaption_Eval[id] for id in sel_num] # list of each group 
        CombineImgCaptionSelected.extend([ImgCaption_Eval[id] for id in sel_num]) 
        
        # Start generating
        for prompt_type in ['prompt_1','prompt_2','prompt_3']:
            Prepared_json = Meta_Text_Grounding(ImgCaptionSelected, prompt_type, dict_ele['category'])
            for ele in Prepared_json:
                run(ele, args, ele['file_name'][:-4], starting_noise) # Generate the image
        
    # Resize and compute FID 
    opt_prompt_type, opt_FID = '', 99999
    for prompt_type in ['prompt_1','prompt_2','prompt_3']:
        temp_FID = FID(CombineImgCaptionSelected, prompt_type, 'generation_box_text') 
        if temp_FID < opt_FID:
            opt_prompt_type, opt_FID = prompt_type, temp_FID
        print('the FID result(Text Grounding) of '+prompt_type + ': ' + str(temp_FID))
        with open('../result_23/FID_TextGrounding_'+prompt_type+'.txt','w') as txt:
            txt.write(str(temp_FID))
    shutil.rmtree('../result_23/generation_box_text/original_dataset')
    shutil.rmtree('../result_23/generation_box_text/generated_imgs')

    # Again, randomly select the twenty images 
    selected_num_list = extract_twenty_img(ImgCaption_Eval) # about the number of selected img of each cat
    CombineImgCaptionSelected = [] # combine the selected_num_list 

    # Random Selected the number of image of each group
    for dict_ele in selected_num_list:
        print("Now is generating the image of the category(Image Grounding): "+ dict_ele['category'])
        sel_num = dict_ele['sel_num']
        ImgCaptionSelected = [ImgCaption_Eval[id] for id in sel_num] # list of each group 
        CombineImgCaptionSelected.extend([ImgCaption_Eval[id] for id in sel_num]) 

        # Start Generating
        Prepared_json = Meta_Image_Grounding(ImgCaptionSelected, opt_prompt_type, dict_ele['category'])
        for ele in Prepared_json:
            run(ele, args, ele['file_name'][:-4], starting_noise) # Generate the image
    FID_result = FID(CombineImgCaptionSelected, opt_prompt_type, 'generation_box_image')
    print('the FID result(Image Grounding) of ' + opt_prompt_type + ": "+str(FID_result))
    with open('../result_23/FID_ImageGrounding_'+opt_prompt_type+'.txt','w') as txt:
        txt.write(str(FID_result))
    shutil.rmtree('../result_23/generation_box_image/original_dataset')
    shutil.rmtree('../result_23/generation_box_image/generated_imgs')
    
                
