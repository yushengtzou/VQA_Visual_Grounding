'''
==========================================================================
 *
 *       Filename:  resizeImage.py
 *
 *    Description:  Resize the images to 224x224  
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
import cv2
import os


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


if __name__ == '__main__':
    # 指定輸入和輸出目錄
    input_directory = '../dataset/Images/origin/train'
    output_directory = '../dataset/Images/resize/train'

    # 呼叫函式以縮放並保存所有影像
    resize_all_images(input_directory, output_directory)




