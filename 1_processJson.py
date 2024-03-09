# Output the Json File
import json
import torch
from PIL import Image
import clip
import os
from efficientnet_pytorch import EfficientNet
# from transformers import Blip2Processor, Blip2ForConditionalGeneration



# 用來抽取影像特徵的函式
def efficientNet():
    model = EfficientNet.from_pretrained('efficientnet-b7')

    # ... image preprocessing as in the classification example ...
    print(img.shape) # torch.Size([1, 3, 224, 224])

    features = model.extract_features(img)
    print(features.shape) # torch.Size([1, 1280, 7, 7])

    image = preprocess(Image.open('../dataset/train/'+img_name)).unsqueeze(0).to(device)




# ---------------------------------------------------------------


# 使用預先訓練好的權重
def GenerateText(file_name, processor, model):
    raw_image = Image.open('hw1_dataset/train/'+file_name).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

# Generate Prompt
def GeneratePrompt(Generated_Text, label, width, height, option):
    stable_text = 'a real photo of ' + Generated_Text + ', ' + label + ', width: ' + str(width) + ', height: '+ str(height) + ', in the aquarium, undersea background, non-grayscale'
    if option == 1:
        option_text = 'camera shake, low HD quality'
    elif option == 2:
        option_text = 'high HD quality, limited color'
    else:
        option_text = 'camera shake, low HD quality, splashes, limited color'
    return (stable_text + option_text) 

def Image_Caption(weight_path, action):
    print('Now is running the blip2 with pretrained model: '+ weight_path)
    # open the annotations
    with open('hw1_dataset/annotations/train.json') as f:
        data = json.load(f)

    # Gather the child categories
    list_categories = [cat["name"] for cat in data['categories']]
    list_images = data['images']
    list_annotations = data['annotations']

    # Load the pretrain weights
    processor = Blip2Processor.from_pretrained(weight_path)
    model = Blip2ForConditionalGeneration.from_pretrained(weight_path, torch_dtype=torch.float16, device_map="auto")

    # output the json
    ImgCation_Dict = []
    for i in range(len(list_images)):
        ele = dict()
        target_id = list_images[i]['id']
        selected_anno = [a for a in list_annotations if a['image_id'] == target_id]
        cat_set = set([j['category_id'] for j in selected_anno])
        bbox_list = [j['bbox'] for j in selected_anno]
        if (len(cat_set) > 1) or (len(cat_set) < 1):
            continue
        if action!='Evaluation':
            if (len(bbox_list) > 6) or (len(bbox_list) < 1):
                continue
        # Extract the proper element
        ele['id'] = list_images[i]['id']
        ele['image'] = list_images[i]['file_name']
        ele['height'] = list_images[i]['height']
        ele['width'] = list_images[i]['width']
        ele['bboxes'] = [[k[0]/ele['width'],k[1]/ele['height'],(k[0]+k[2])/ele['width'],(k[1]+k[3])/ele['height']] for k in bbox_list]
        ele['label'] = list_categories[list(cat_set)[0]]
        ele['generated_text'] = GenerateText(ele['image'], processor, model)
        if ele['label'] == 'puffin':
            ele['generated_text'] = (ele['generated_text'].replace('penguin', 'puffin')).replace('bird','puffin')
        ele['prompt_1'] = GeneratePrompt(ele['generated_text'], ele['label'], ele['width'], ele['height'], 1)  
        ele['prompt_2'] = GeneratePrompt(ele['generated_text'], ele['label'], ele['width'], ele['height'], 2) 
        ele['prompt_3'] = GeneratePrompt(ele['generated_text'], ele['label'], ele['width'], ele['height'], 3) 
        ImgCation_Dict.append(ele)
    return ImgCation_Dict

def Calculate_Performance(json_obj):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    score = 0
    for ele in json_obj:
        img_name = ele['image']
        image = preprocess(Image.open('hw1_dataset/train/'+img_name)).unsqueeze(0).to(device)
        text = clip.tokenize([ele['generated_text']]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image,text)
        score += 2.5 * max(torch.nn.functional.cosine_similarity(image_features, text_features).item(), 0)
    return score/len(json_obj)

if __name__ == '__main__':
    
    os.makedirs('result_1', exist_ok=True)
    # Write the json file for FID Evaluation
    opt_weightname, opt_weightpath, opt_score, opt_json, = '', '',  0, None
    compare_dict = dict()
    for weight_name in ['flan-t5-xl', 'opt-6.7b-coco', 'opt-6.7b', 'opt-2.7b']:
        weight_path = 'Salesforce/blip2-' + weight_name
        output_json = Image_Caption(weight_path, 'Evaluation')
        score = Calculate_Performance(output_json)
        print('The score(performance) of ' + weight_path + ' is: ' + str(score))
        compare_dict[weight_name]=score
        if score > opt_score:
            opt_score = score
            opt_json = output_json
            opt_weightname = weight_name
            opt_weightpath = weight_path
    print("The pretrained model I selected among the 4 models is: " + opt_weightpath)
    file_name = 'ImageCaption_Evaluation.json'
    print("The json file(For FID Evaluation) is stored in: "+ file_name)
    with open('result_1/'+file_name, 'w') as f:
        json.dump(opt_json, f, indent=2)

    # write the results of the four pre-trained models
    with open('result_1/pretrained_compare_results.txt','w') as ff:
        json.dump(compare_dict, ff, indent=2)

    # Write the json file for Data Augmentation
    output_json = Image_Caption(opt_weightpath, '')
    file_name = 'ImageCaption_DataAugment.json'
    print("The json file(For Data Augmentation) is stored in: "+ file_name)
    with open('result_1/'+file_name, 'w') as f:
        json.dump(opt_json, f, indent=2)


