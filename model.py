'''
==========================================================================
 *
 *       Filename:  model.py
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
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 定義影像-文本的編碼器的函式 
def encoder(image, question, answers):
    features = []
    image = preprocess(image).unsqueeze(0).to(device)
    question = clip.tokenize(question).to(device)
    answers = clip.tokenize(answers).to(device)

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # answers = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # 所有計算出的 tensor 的 requires_grad 都自動設置為 False
    with torch.no_grad():
        image_features = model.encode_image(image)
        question_features = model.encode_text(question)
        answers_features = model.encode_text(answers)

        features.append(image_features)
        features.append(question_features)
        features.append(answers_features)
    return features


# 編碼影像-問題-答案的函式 
def encodeImageQuestionAnswers(inputJsonPath, inputImageDirectoryPath):
    with open(inputJsonPath, 'r') as file:
        data = json.load(file)

    features = []

    # 遍歷 data 中的每一個 dict 
    for imageFileName, info in data.items():
        imagePath = os.path.join(inputImageDirectoryPath, imageFileName)
        # 讀取影像
        img = cv2.imread(imagePath)
        question = info["question"]
        answers = info["answers"]

        # 呼叫編碼器函式編碼
        features.append(encoder(img, question, answers))
        print(f"已抽取特徵並保存：{imageFileName}")

    return features


if __name__ == '__main__':
    # 指定輸入和輸出目錄
    inputDirectory = '../dataset/Images/origin/train'
    outputDirectory = '../dataset/Images/resize/train'

    # 呼叫函式以縮放並保存所有影像
    encoder(inputDirectory, outputDirectory)


