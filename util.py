'''
==========================================================================
 *
 *       Filename:  util.py
 *
 *    Description:  Read JSON file to process data  
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


import os
import json


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






    # # 遍歷 imageAnnotationRecord 中的每一個 image 
    # # 以篩選出只有一個 labelRecord 的 image
    # # 並將該 image 的相關記錄存到 imageWith1Category
    # for image, record in imageAnnotationRecord.items():
    #     if len(record["labels"].keys()) == 1:
    #         imageWith1Category[image] = record 
    #
    # # 遍歷 data['images'] 中的每一個 image 
    # for image in data['images']:
    #     record = {} # 儲存影像相關記錄的字典
    #
    #     imageName = image['file_name']
    #     imageId = image['id']
    #     height = image['height']
    #     width = image['width']
    #
    #     labelRecord = {} # 儲存影像標註記錄的字典
    #     # 對於 labelRecord 的特別處理
    #     # 遍歷 data['annotations'] 中的每一個 annotation 
    #     for annotation in data['annotations']:
    #         if imageId == annotation['image_id']:
    #             categoryId = annotation['category_id']
    #             labelName = label[categoryId]
    #             bbox = annotation['bbox']
    #
    #             # 標準化 bbox 的值
    #             x_min, y_min, bbox_width, bbox_height = bbox
    #             x_max = x_min + bbox_width
    #             y_max = y_min + bbox_height
    #
    #             normalizedBbox = [
    #                 round(x_min / width, 2),
    #                 round(y_min / height, 2),
    #                 round(x_max / width, 2),
    #                 round(y_max / height, 2)
    #             ]
    #
    #             # Check if labelName already in labelRecord
    #             if labelName not in labelRecord:
    #                 labelRecord[labelName] = {"bboxs": []}
    #             labelRecord[labelName]["bboxs"].append(normalizedBbox)
    #
    #
    #     # 將相關記錄儲存到 record 中
    #     record["height"] = height 
    #     record["width"] = width
    #     record["labels"] = labelRecord 
    #     record["generated_text"] = "" 
    #     record["prompt_w_label"] = "" 
    #     record["prompt_w_suffix"] = "" 
    #     # 以 imageName 為 key
    #     imageAnnotationRecord[imageName] = record 
    #
    # # 一個檢查點
    # # for key, value in imageAnnotationRecord.items():
    # #     print('Image: ', key, 'Record: ', value)
    #
    # # 遍歷 imageAnnotationRecord 中的每一個 image 
    # # 以篩選出只有一個 labelRecord 的 image
    # # 並將該 image 的相關記錄存到 imageWith1Category
    # for image, record in imageAnnotationRecord.items():
    #     if len(record["labels"].keys()) == 1:
    #         imageWith1Category[image] = record 
    #
    # # 一個檢查點
    # # for key, value in imageWith1Category.items():
    # #     print('Image: ', key, 'Record: ', value)
    #
    # # 遍歷 imageWith1Category 中的每一個 image
    # # 以篩選出只有一個 bbox 的 image
    # # 並將該 image 的相關記錄存到 imageWith1Bbox
    # # 結果只有 96 個 
    # for image, record in imageWith1Category.items():
    #     values = list(record["labels"].values())
    #     bboxs = values[0]["bboxs"]  
    #     if len(bboxs) == 1:
    #         # imageWith1Bbox[image] = record
    #         imageWith123Bbox[image] = record
    #
    # # 遍歷 imageWith1Category 中的每一個 image
    # # 以篩選出只有 2 個 bbox 的 image
    # # 並將該 image 的相關記錄存到 imageWith2Bbox
    # for image, record in imageWith1Category.items():
    #     values = list(record["labels"].values())
    #     bboxs = values[0]["bboxs"]  
    #     if len(bboxs) == 2:
    #         # imageWith2Bbox[image] = record
    #         imageWith123Bbox[image] = record
    #
    # # 遍歷 imageWith1Category 中的每一個 image
    # # 以篩選出只有 3 個 bbox 的 image
    # # 並將該 image 的相關記錄存到 imageWith3Bbox
    # for image, record in imageWith1Category.items():
    #     values = list(record["labels"].values())
    #     bboxs = values[0]["bboxs"]  
    #     if len(bboxs) == 3:
    #         # imageWith3Bbox[image] = record
    #         imageWith123Bbox[image] = record
    #
    # # 一個檢查點
    # # for key, value in imageWith123Bbox.items():
    # #     print('Image: ', key, 'Record: ', value)
    #
    # processImageWith123Bbox(imageWith123Bbox, imageWith1Category)
    
