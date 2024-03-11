'''
==========================================================================
 *
 *       Filename:  util.py
 *
 *    Description:  Resize the images to 224x224,  
                    Read JSON file to process data  
 *
 *        Version:  1.0
 *        Created:  2024/03/09
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
import json
import cv2


def resize_all_images(input_directory, output_directory):
    # 確保輸出目錄存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍歷輸入目錄中的所有文件
    for filename in os.listdir(input_directory):
        # 構建完整的文件路徑
        input_path = os.path.join(input_directory, filename)

        # 檢查文件是否為影像（為了簡單起見，這裡只檢查檔案副檔名）
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # 讀取影像
            img = cv2.imread(input_path)

            # 將影像縮放到 224x224
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            # 構建輸出文件路徑
            output_path = os.path.join(output_directory, filename)

            # 保存縮放後的影像
            cv2.imwrite(output_path, img)

            print(f"已縮放並保存：{output_path}")


# 處理 json 檔資料的函式 
def process_json(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # 遍歷 data 中的每一個 dict 
    for image_file_name, info in data.items():
        origin_height = info["height"]
        origin_width = info["width"]
        info["height"] = 224
        info["width"] = 224

        for point in info["answer_grounding"]:
            point["x"] = point["x"] * 224 / originWidth
            point["y"] = point["y"] * 224 / originHeight

        print(f"已處理並保存：{image_file_name}")

    # 將處理後的數據寫回到一個新的 JSON 檔案中
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    # 指定輸入和輸出目錄
    input_directory = '../dataset/Images/origin/train'
    output_directory = '../dataset/Images/resize/train'

    # 呼叫函式以縮放並保存所有影像
    resize_all_images(input_directory, output_directory)


