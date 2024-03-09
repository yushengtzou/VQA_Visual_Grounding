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


# 引入相關副程式
import resizeImage
import os
from util import processJson, processImageWith123Bbox


def main():
    # ------------------- 相關路徑設定 -------------------

    # 設定 json 路徑
    file_path = '../dataset/annotations/annotations/train_grounding.json'

    # ------------------- 呼叫函式 -------------------

    # 讀取 json 檔，並篩選出只有一個 category，每個 category
    # 有多個 bbox 的 image 和其記錄，再依照 category
    # 各存成一個 json 檔，將這些 json 檔存入新建的 ./category 資料夾
    processJson(file_path)


if __name__ == '__main__':
    main()
