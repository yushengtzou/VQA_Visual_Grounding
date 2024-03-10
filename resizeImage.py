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


def resizeAllImages(inputDirectory, outputDirectory):
    # 確保輸出目錄存在
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # 遍歷輸入目錄中的所有文件
    for filename in os.listdir(inputDirectory):
        # 構建完整的文件路徑
        inputPath = os.path.join(inputDirectory, filename)

        # 檢查文件是否為影像（為了簡單起見，這裡只檢查檔案副檔名）
        if inputPath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # 讀取影像
            img = cv2.imread(inputPath)

            # 將影像縮放到 224x224
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            # 構建輸出文件路徑
            outputPath = os.path.join(outputDirectory, filename)

            # 保存縮放後的影像
            cv2.imwrite(outputPath, img)

            print(f"已縮放並保存：{outputPath}")


if __name__ == '__main__':
    # 指定輸入和輸出目錄
    inputDirectory = '../dataset/Images/origin/train'
    outputDirectory = '../dataset/Images/resize/train'

    # 呼叫函式以縮放並保存所有影像
    resizeAllImages(inputDirectory, outputDirectory)




