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


# 從 imageWith123Bbox 篩選出 label 是何物種的 img
# 並個別存入各物種的字典中，再存成該物種的 json 檔
# 將新建的 json 檔 存入新建的 category 資料夾中
def processImageWith123Bbox(imageWith123Bbox, imageWith1Category):
    # 儲存影像標註類別是 某一物種 的字典
    fishImageRecord = {}
    jellyfishImageRecord = {}
    penguinImageRecord = {}
    puffinImageRecord = {}
    sharkImageRecord = {}
    starfishImageRecord = {}
    stingrayImageRecord = {}

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 fish 的 img
    # 並將該 image 的相關記錄存到 fishImageRecord
    for image, record in imageWith123Bbox.items():
        if "fish" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', fish, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])
            print(record["prompt_w_label"])
            print(record["prompt_w_suffix"])

            fishImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 jellyfish 的 img
    # 並將該 image 的相關記錄存到 jellyfishImageRecord
    for image, record in imageWith1Category.items():
        if "jellyfish" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', jellyfish, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            jellyfishImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 penguin 的 img
    # 並將該 image 的相關記錄存到 penguinImageRecord
    for image, record in imageWith123Bbox.items():
        if "penguin" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', penguin, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            penguinImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 puffin 的 img
    # 並將該 image 的相關記錄存到 puffinImageRecord
    for image, record in imageWith123Bbox.items():
        if "puffin" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', puffin, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            puffinImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 shark 的 img
    # 並將該 image 的相關記錄存到 sharkImageRecord
    for image, record in imageWith123Bbox.items():
        if "shark" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', shark, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            sharkImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 starfish 的 img
    # 並將該 image 的相關記錄存到 starfishImageRecord
    for image, record in imageWith123Bbox.items():
        if "starfish" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', starfish, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            starfishImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break

    counter = 0
    max_records = 20
    # 遍歷 imageWith123Bbox 中的每一個 image
    # 以篩選出 label 是 stingray 的 img
    # 並將該 image 的相關記錄存到 stingrayImageRecord
    for image, record in imageWith123Bbox.items():
        if "stingray" in record["labels"]:
            # 設定圖片路徑
            image_path = '/mnt/lab/2.course/112-1/CVPDL_hw/hw3/hw1_dataset/train/' + image
            # 使用 BLIP2 圖生文字
            generated_text = image2Text27b(image_path, cache_dir)
            record["generated_text"] = generated_text
            record["prompt_w_label"] = generated_text + ', stingray, height: '+ str(record["height"]) + ', width: ' + str(record["width"])
            record["prompt_w_suffix"] = record["prompt_w_label"] + ', ocean, undersea background, HD quality, high detailed'

            print(record["generated_text"])

            stingrayImageRecord[image] = record
            counter += 1
            if counter == max_records:
                break


    # 建立一個資料夾 'category' 若他不存在
    if not os.path.exists('category'):
        os.makedirs('category')

    # 將 json 檔 存入資料夾 'category' 的函式
    def save_json(filename, data):
        with open(os.path.join('category', filename), 'w') as file:
            json.dump(data, file, indent=4)

    # 將各 record 字典 存成 json 檔
    save_json('fish.json', fishImageRecord)
    save_json('jellyfish.json', jellyfishImageRecord)
    save_json('penguin.json', penguinImageRecord)
    save_json('puffin.json', puffinImageRecord)
    save_json('shark.json', sharkImageRecord)
    save_json('starfish.json', starfishImageRecord)
    save_json('stingray.json', stingrayImageRecord)

    # 一個檢查點
    for key, value in jellyfishImageRecord.items():
        print('Image: ', key, 'Record: ', value)


# 處理 json 檔資料的函式 
def processJson(inputFilePath, outputFilePath):
    with open(inputFilePath, 'r') as file:
        data = json.load(file)

    # 儲存影像 answer_grounding 座標們的串列
    answerGroundingList = []
    # 儲存影像 answer_grounding 座標的字典
    answerGroundingCoord = {}

    # 遍歷 data 中的每一個 dict 
    for imageFileName, info in data.items():
        originHeight = info["height"]
        originWidth = info["width"]
        info["height"] = 224
        info["width"] = 224

        for point in info["answer_grounding"]:
            point["x"] = point["x"] * 224 / originWidth
            point["y"] = point["y"] * 224 / originHeight

        print(f"已處理並保存：{imageFileName}")

    # 將處理後的數據寫回到一個新的 JSON 檔案中
    with open(outputFilePath, 'w') as outfile:
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
    
