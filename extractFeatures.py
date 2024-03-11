'''
==========================================================================
 *
 *       Filename:  extractFeatures.py
 *
 *    Description:  Use CLIP model API
 *
 *        Version:  1.0
 *        Created:  2024/03/10
 *       Revision:  none
 *       Compiler:  
 *
 *         Author:  鄒雨笙 
 *   Organization:  
 *
 * ==========================================================================
'''


# 引入相關模組
import os
import cv2
import torch
import json 
import clip
import numpy as np
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 定義影像-文本的編碼器的函式 
def encoder(image, question, answers):
    features = []
    image = preprocess(image).unsqueeze(0).to(device)
    question = clip.tokenize(question).to(device)
    answers = clip.tokenize(answers).to(device)

    # 所有計算出的 tensor 的 requires_grad 都自動設置為 False
    with torch.no_grad():
        image_features = model.encode_image(image)
        question_features = model.encode_text(question)
        answers_features = model.encode_text(answers)

        print(f"Type of image_features: {type(image_features)}, Shape: {image_features.shape}")
        print(f"Type of question_features: {type(question_features)}, Shape: {question_features.shape}")
        print(f"Type of answers_features: {type(answers_features)}, Shape: {answers_features.shape}")

        features.append(image_features)
        features.append(question_features)
        features.append(answers_features)
    return features


# 編碼影像-問題-答案的函式 
def encode_image_question_answers(input_json_path, input_image_directory_path):
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    features_list = []

    # 遍歷 data 中的每一個 dict 
    for image_file_name, info in data.items():
        image_path = os.path.join(input_image_directory_path, image_file_name)
        # 讀取影像
        img = cv2.imread(image_path)
        # Convert the image from BGR (OpenCV) to RGB 
        # and then to a PIL Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        question = info["question"]
        answers = info["answers"]

        # 呼叫編碼器函式編碼
        features_list.append(encoder(img, question, answers))
        print(f"已抽取特徵：{image_file_name}")

    return features_list


# 處理影像-問題-答案特徵的函式 
def process_extracted_features(features_list):
    # 遍歷 featuresList 中的每一個 List
    for features in features_list:
        image_features = features[0]
        question_features = features[1]
        answers_features = features[2]
        # Concatenate questionFeatures and answersFeatures to a vector 







# if __name__ == '__main__':
    # 指定輸入和輸出目錄
    # inputDirectory = '../dataset/Images/origin/train'
    # outputDirectory = '../dataset/Images/resize/train'

    # 呼叫函式以縮放並保存所有影像
    # encoder(inputDirectory, outputDirectory)


