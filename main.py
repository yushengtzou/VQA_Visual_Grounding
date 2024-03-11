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
input_image_file_path = '../dataset/Images/origin/train'
output_image_file_path = '../dataset/Images/resize/train'

# 指定 JSON 檔案輸入和輸出的路徑(處理 json 檔案用)
input_json_file_path = '../dataset/annotations/annotations/train_grounding.json'
output_json_file_path = '../dataset/annotations/annotationsProcessed/train_grounding_processed.json'

# 指定 JSON 檔案輸入和輸出的路徑(抽取 image-question-answer 特徵用)
input_json_file_path_for_feature = '../dataset/annotations/annotationsProcessed/train_grounding_processed.json'
resized_image_file_path = '../dataset/Images/resize/train'

# ==================================================================


# 主程式
def main():
    # ------------------- 呼叫副程式的函式 -------------------

    # 呼叫 resize_all_images() 函式以縮放並保存所有影像
    # resizeImage.resize_all_images(input_image_file_path, output_image_file_path)

    # 呼叫 process_json() 函式，處理 JSON 檔案並保存結果
    # util.process_json(input_json_file_path, output_json_file_path)

    # 呼叫 encode_image_question_answers() 函式，以抽取特徵
    features = extractFeatures.encode_image_question_answers(input_json_file_path_for_feature, resized_image_file_path)

    # 呼叫 clustering_features() 函式，來視覺化特徵空間
    extractFeatures.clustering_features(features)


if __name__ == '__main__':
    main()



