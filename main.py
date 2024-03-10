'''  ==========================================================================
 *
 *       Filename:  main.py
 *
 *    Description:  此項目的主程式  
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


# 引入副程式
import resizeImage
import util
import extractFeatures
import model 


# ============================ 路徑設定 ============================

# 指定 resize image 檔案輸入和輸出的路徑(resize image 用)
inputImageFilePath = '../dataset/Images/origin/train'
outputImageFilePath = '../dataset/Images/resize/train'

# 指定 JSON 檔案輸入和輸出的路徑(處理 json 檔案用)
inputJsonFilePath = '../dataset/annotations/annotations/train_grounding.json'
outputJsonFilePath = '../dataset/annotations/annotationsProcessed/train_grounding_processed.json'

# 指定 JSON 檔案輸入和輸出的路徑(抽取 image-question-answer 特徵用)
inputJsonFilePathForFeature = '../dataset/annotations/annotationsProcessed/train_grounding_processed.json'
resizedImageFilePath = '../dataset/Images/resize/train'

# ==================================================================


# 主程式
def main():
    # ------------------- 呼叫副程式的函式 -------------------

    # 呼叫 resizeAllImages() 函式以縮放並保存所有影像
    # resizeImage.resizeAllImages(inputImageFilePath, outputImageFilePath)

    # 呼叫 processJson() 函式，處理 JSON 檔案並保存結果
    # util.processJson(inputJsonFilePath, outputJsonFilePath)

    # 呼叫 encodeImageQuestionAnswers() 函式，以抽取特徵
    features = extractFeatures.encodeImageQuestionAnswers(inputJsonFilePathForFeature, resizedImageFilePath)

    # 呼叫 clusteringFeatures() 函式，來視覺化特徵空間
    extractFeatures.clusteringFeatures(features)


if __name__ == '__main__':
    main()



