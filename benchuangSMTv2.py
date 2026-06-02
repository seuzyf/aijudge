import cv2
import math
# import resizeimg  # <--- 已注释掉：不再需要 resizeimg
import os
import xml.etree.ElementTree as ET
from paddleocr import DocImgOrientationClassification
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget
import time
import pandas as pd
import shutil
import logging
import numpy as np
from datetime import datetime
import csv
import io

os.environ['PADDLE_USE_MKLDNN'] = '0'

model = YOLO("./model/benchuang.pt")
model.eval()
modelForeign = YOLO("./model/feiliao.pt")
modelForeign.eval()
logging.info("载入模型：奔创")
ocr_model = DocImgOrientationClassification(model_dir="./model/PP-LCNet_x1_0_doc_ori")
logging.basicConfig(
    filename='check.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置
)
logging.info("载入OCR模型完成")
class_names = ['ChipR', 'SOP', 'SOT23', 'QFP']
DISPLAY_FOLDER = "display"
os.makedirs(DISPLAY_FOLDER, exist_ok=True)
disp = 1


def copy_to_display_folder(image_path, result):
    """将图片按 {result}_{count}.jpg 命名拷贝到 display 文件夹"""
    global disp
    disp += 1
    new_name = f"{result}_{disp}.jpg"
    dest_path = os.path.join(DISPLAY_FOLDER, new_name)
    shutil.copy(image_path, dest_path)

def read_threshold_from_config(config_path='config.txt'):
    default_okrange = 0.5
    default_collect = 0
    default_lag = 1.0  # 新增 lag 默认值，使用浮点数以兼容小数秒
    
    okrange_found = False
    collect_found = False
    lag_found = False  # 新增 lag 查找标志
    
    # 初始化变量，防止文件存在但某项缺失时引发 UnboundLocalError
    okrange_value = default_okrange
    collect_value = default_collect
    lag_value = default_lag

    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f"okRange={default_okrange}\n")
                f.write(f"collect={default_collect}\n")
                f.write(f"lag={default_lag}\n")
            return default_okrange, default_collect, default_lag
            
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                if line.startswith("okRange"):
                    _, val = line.split("=")
                    okrange_value = float(val)
                    okrange_found = True
                elif line.startswith("collect"):
                    _, val = line.split("=")
                    collect_value = int(val)
                    collect_found = True
                elif line.startswith("lag"):
                    _, val = line.split("=")
                    lag_value = float(val)
                    lag_found = True
                    
        with open(config_path, 'a') as f:
            if not okrange_found:
                f.write(f"okRange={default_okrange}\n")
            if not collect_found:
                f.write(f"collect={default_collect}\n")
            if not lag_found:
                f.write(f"lag={default_lag}\n")
                
        return okrange_value, collect_value, lag_value
    except Exception as e:
        logging.error(f"配置读取失败: {e}")
        return default_okrange, default_collect, default_lag

def copy_image_with_suffix(src_path, dest_folder, ngtype, result):
    """将图片按 {原文件名}_{NG_NAME}_{FLAG}.jpg 命名拷贝到目标文件夹"""
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    # 替换斜杠为下划线，防止路径问题
    ngtype_safe = ngtype.replace('/', '_')
    base_filename = f"{base_name}_{ngtype_safe}_{result}{ext}"
    dest_path = os.path.join(dest_folder, base_filename)
    counter = 1
    while os.path.exists(dest_path):
        new_filename = f"{base_name}_{ngtype_safe}_{result}_{counter}{ext}"
        dest_path = os.path.join(dest_folder, new_filename)
        counter += 1
    shutil.copy(src_path, dest_path)
    logging.info(f"图片已复制到 {dest_path}")

def write_to_history(task_order, program_name, board_code, image_name, ngtype, result):
    csv_file = 'history.csv'
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = [
        task_order,
        program_name,
        str(board_code),
        image_name,
        ngtype,
        result,
        current_date
    ]
    expected_header = ['任务令', '程序名', '单板条码', '图片名', '缺陷类型', '结果', '日期']
    # 如果文件不存在，直接写入表头+数据
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)
            writer.writerow(new_row)
        return
    # 文件存在，检查是否有表头
    with open(csv_file, mode='r', newline='', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()
    # 如果第一行不是预期的表头，则备份原文件并插入表头
    if first_line != ','.join(expected_header):
        backup_file = csv_file + '.bak'
        os.rename(csv_file, backup_file)
        with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as fout:
            writer = csv.writer(fout)
            writer.writerow(expected_header)
            # 将旧文件内容复制到新文件中
            with open(backup_file, mode='r', newline='', encoding='utf-8-sig') as fin:
                for line in fin:
                    fout.write(line)
        print(f"【警告】检测到文件无有效表头，已自动补全，并从 {backup_file} 恢复数据")
    # 追加写入新行
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)
    print(f"【write_to_history】已写入一行数据到 {csv_file}")

# 解析XML文件获取角度的函数 
def get_angle_from_xml(xml_path):
    """
    从XML文件中查找第一个<PartData>下的<Roi><a>值。
    如果未找到<a>，返回None。
    特殊处理：270 返回 90，90 返回 270。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 查找 PartData 标签
        part_data = root.find('PartData')
        if part_data is not None:
            # 在 PartData 下查找 Roi/a
            roi_elem = part_data.find('Roi')
            if roi_elem is not None:
                angle_elem = roi_elem.find('a')
                if angle_elem is not None and angle_elem.text is not None:
                    try:
                        angle_value = float(angle_elem.text)
                        # 特殊角度转换
                        if angle_value == 270.0:
                            angle_value = 90.0
                        elif angle_value == 90.0:
                            angle_value = 270.0
                        print(f"从XML获取角度并转换后: {angle_value}")
                        return angle_value
                    except ValueError:
                        logging.warning(f"XML文件 {xml_path} 中的angle值 '{angle_elem.text}' 无法转换为浮点数。")
                        return None
                else:
                    logging.warning(f"在XML文件 {xml_path} 的 Roi 中，未找到<a>标签。")
                    return None
            else:
                logging.warning(f"在XML文件 {xml_path} 的 PartData 中，未找到<Roi>标签。")
                return None
        else:
            logging.warning(f"在XML文件 {xml_path} 中，未找到<PartData>标签。")
            return None

    except ET.ParseError as e:
        logging.error(f"解析XML文件 {xml_path} 失败: {e}")
        return None
    except Exception as e:
        logging.error(f"读取或处理XML文件 {xml_path} 时发生未知错误: {e}")
        return None


def process_all_files(check, directory, xmlPath):
    logging.info("当前复判xml读取路径：%s", xmlPath)
    parent_dir = os.path.dirname(directory)
    historyPath = os.path.join(parent_dir, 'history')
    okPath = os.path.join(historyPath, 'OK')
    ngPath = os.path.join(historyPath, 'NG')
    resPath = os.path.join(parent_dir, 'AI')
    os.makedirs(historyPath, exist_ok=True)
    os.makedirs(okPath, exist_ok=True)
    os.makedirs(ngPath, exist_ok=True)
    logging.info("当前复判结果归档路径：%s", historyPath)
    # 初始化全局变量
    global disp
    disp = 1
    DISPLAY_FOLDER = "display"
    os.makedirs(DISPLAY_FOLDER, exist_ok=True)
    # 接收新增的 lag 参数
    okrange, collect, lag = read_threshold_from_config()


    while check:
        items = os.listdir(directory)
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        if not folders:
            time.sleep(2)
            continue

        #  获取当前处理的文件夹名
        current_folder_name = folders[0]
        checkPath = os.path.join(directory, current_folder_name) # 使用 current_folder_name
        imagePath = os.path.join(checkPath, 'NGPartImage')

        # --- 新增的延时逻辑 ---
        logging.info(f"扫描到复判文件夹: {current_folder_name}，等待 {lag} 秒以确保 CSV 文件输出完全...")
        time.sleep(lag)
        # ----------------------


        for filename in os.listdir(checkPath):
            if not filename.endswith(".csv"):
                continue
            
            # 获取 xmlPath 下的所有子文件夹（第一层，随机名）
            level1_subfolders = [f for f in os.listdir(xmlPath) if os.path.isdir(os.path.join(xmlPath, f))]

            if not level1_subfolders:
                logging.warning(f"XML路径下没有子文件夹: {xmlPath}")
                xml_subfolder_path = None
            else:
                # 按修改时间排序，从新到旧
                level1_subfolders.sort(key=lambda x: os.path.getmtime(os.path.join(xmlPath, x)), reverse=True)
                
                xml_subfolder_path = None
                for folder in level1_subfolders:
                    candidate_path = os.path.join(xmlPath, folder, current_folder_name)
                    if os.path.exists(candidate_path):
                        xml_subfolder_path = candidate_path
                        logging.info(f"找到XML子文件夹: {xml_subfolder_path}")
                        break
                
                if xml_subfolder_path is None:
                    logging.warning(f"在所有候选路径中都未找到XML子文件夹: {current_folder_name}")

            csvPath = os.path.join(checkPath, filename)
            file_name, _ = os.path.splitext(filename)
            # 不再创建子文件夹，直接使用OK和NG文件夹
            os.makedirs(okPath, exist_ok=True)
            os.makedirs(ngPath, exist_ok=True)
            try:
                with open(csvPath, 'rb') as f:
                    content_bytes = f.read()
                
                # 首先尝试 utf-8-sig (处理带有BOM的UTF-8文件)
                try:
                    decoded_content = content_bytes.decode('utf-8-sig')
                    logging.info("'utf-8-sig' 解码成功。")
                except UnicodeDecodeError:
                    # 如果失败，则使用 gbk 并忽略无法解码的字符
                    decoded_content = content_bytes.decode('gbk', errors='ignore')
                    logging.info("'gbk' (忽略错误) 解码完成。")

                # 将解码后的干净字符串交给Pandas处理
                df = pd.read_csv(io.StringIO(decoded_content))

            except Exception as e:
                logging.error(f"读取CSV文件 {csvPath} 的所有方法均失败: {e}", exc_info=True)
                continue # 跳过这个有严重问题的文件，继续循环
            logging.info("CSV读取成功。")
            
            # -------------------------------------------------------------------------
            # <--- 修改点 1: 删除了 resize = 640 和 resizeimg 处理逻辑
            # resize = 640
            # resizeimg.process_images_in_folder(imagePath, imagePath, new_shape=(resize, resize))
            # center_x, center_y = resize / 2, resize / 2
            # -------------------------------------------------------------------------

            for index, row in df.iterrows():
                # 从CSV中提取必要信息
                program_name = row.get('JOBNAME', '')   # 程序名
                board_code = row.get('ARRAY_BARCODE', '')  # 单板条码
                ngtype = row.get('NG_NAME', '').lower()         # 缺陷类型
                file_name_part = f"{row.iloc[4]}@{row.iloc[5]}"  # 使用iloc避免FutureWarning
                # 定义可能的后缀列表（按优先级排序）
                suffix_list = [
                    "_AC.jpg",    # 优先尝试带_AC后缀的
                    ".jpg",       # 其次尝试不带后缀的
                    "_ac.jpg",    # 小写_ac后缀
                    "_AC.JPG",    # 大写扩展名
                    ".JPG",       # 大写扩展名不带后缀
                    "_AC.png",    # 其他可能的格式
                    ".png"
                ]
                # 尝试查找存在的文件
                image_path = None
                for suffix in suffix_list:
                    temp_path = os.path.join(imagePath, f"{file_name_part}{suffix}")
                    if os.path.exists(temp_path):
                        image_path = temp_path
                        break
                if not image_path:
                    # 如果所有后缀都尝试过了仍未找到
                    default_path = os.path.join(imagePath, f"{file_name_part}_AC.jpg")  # 默认路径用于记录
                    logging.error(f"图片未找到: {default_path} (尝试了所有后缀: {suffix_list})")
                    print(f"图片未找到: {default_path} (尝试了所有后缀: {suffix_list})")
                    df.at[index, 'FLAG'] = 'AING'
                    continue
                
                imgname = os.path.splitext(os.path.basename(image_path))[0]

                # ---------------------------------------------------------------------
                # <--- 修改点 2: 动态获取图片的宽和高，并计算当前图片的中心点
                # 因为不再统一Resize，所以需要对每张图获取实际尺寸
                # ---------------------------------------------------------------------
                img_for_dim = cv2.imread(image_path)
                if img_for_dim is None:
                    logging.error(f"OpenCV无法读取图像: {image_path}")
                    df.at[index, 'FLAG'] = 'AING'
                    continue
                h_img, w_img = img_for_dim.shape[:2]
                center_x, center_y = w_img / 2.0, h_img / 2.0
                # ---------------------------------------------------------------------

                if ngtype == 'ocr' or ngtype == 'wrong':
                    logging.info(f"处理 ngtype 为 'OCR' 的图片: {image_path}")
                    res = "NG"  # 默认结果
                    checkresult = "AING"  # 默认结果

                    try:
                        # 1. 加载图像 (上面已经读取过一次 img_for_dim，可以直接复用，但为了逻辑清晰暂保持原有流程或直接使用)
                        image = img_for_dim # 复用已读取的图片
                        
                        # 2. 裁剪画面正中心的1/3区域（保持原图像宽高比）
                        h, w = image.shape[:2]
                        
                        # 计算中心1/3区域的尺寸
                        new_w = w // 3
                        new_h = h // 3
                        
                        # 计算裁剪区域的起始点
                        start_x = (w - new_w) // 2
                        start_y = (h - new_h) // 2
                        
                        # 裁剪中心区域
                        cropped_image = image[start_y:start_y + new_h, start_x:start_x + new_w]
                        
                        # 3. 转换为灰度图
                        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                        # 4. 使用 convertScaleAbs 调整对比度和亮度
                        enhanced_image = cv2.convertScaleAbs(gray_image, alpha=3, beta=0)

                        # 5. 将处理后的图像保存为临时路径供模型预测使用
                        temp_image_path = image_path.replace(".jpg", "_enhanced.jpg")
                        cv2.imwrite(temp_image_path, enhanced_image)

                        # 4. 调用 OCR 模型进行预测
                        ocr_output = ocr_model.predict(temp_image_path, batch_size=1)
                        ocr_angle = None
                        ocr_score = None
                        # 删除临时图片
                        # os.remove(temp_image_path)
                        if ocr_output and len(ocr_output) > 0:
                            first_result = ocr_output[0]

                            if isinstance(first_result, dict):
                                label_names = first_result.get('label_names', [])
                                scores = first_result.get('scores', [])

                                if label_names and len(label_names) > 0 and scores and len(scores) > 0:
                                    ocr_angle_str = label_names[0]
                                    ocr_angle = float(ocr_angle_str)
                                    ocr_score = float(scores[0])
                                    logging.info(f"OCR预测角度: {ocr_angle}°, 置信度: {ocr_score}")
                                    # 如果置信度小于okrange，则角度无效
                                    if ocr_score < okrange:
                                        ocr_angle = None
                                        logging.warning("OCR置信度低于阈值，忽略预测角度。复判结果: NG")

                        if ocr_angle is not None:
                            # 2. *** 构造 XML 文件路径 - 使用 xml_subfolder_path ***
                            xml_filename = os.path.splitext(os.path.basename(image_path))[0] + ".xml" # 基于图片名构造XML名
                            xml_path = os.path.join(xml_subfolder_path, xml_filename) # 新路径
                            # *** 检查 XML 文件是否存在 ***
                            if os.path.exists(xml_path):
                                # 3. 解析XML文件获取角度 (修改后的函数)
                                xml_angle = get_angle_from_xml(xml_path) # 不再传递图片名
                                if xml_angle is not None:
                                    # 4. 简化角度比较逻辑 (只考虑 0, 90, 180, 270)
                                    # 定义有效角度列表
                                    valid_angles = [0.0, 90.0, 180.0, 270.0]
                                    # 检查 OCR 和 XML 角度是否都在有效列表中
                                    if ocr_angle in valid_angles and xml_angle in valid_angles:
                                        # 直接比较数值是否相等
                                        if ocr_angle == xml_angle:
                                            res = "OK"
                                            checkresult = "AIOK"
                                            logging.info(f"OCR角度({ocr_angle}°) 与 XML角度({xml_angle}°) 一致。复判结果: OK")
                                        else:
                                            res = "NG"
                                            checkresult = "AING"
                                            logging.info(f"OCR角度({ocr_angle}°) 与 XML角度({xml_angle}°) 不一致。复判结果: NG")
                    except Exception as e:
                         logging.error(f"处理 ngtype 'OCR' 时发生异常: {e}", exc_info=True) # exc_info=True 记录堆栈
                         res = "NG"
                         checkresult = "AING"
                    # 写入历史记录
                    write_to_history(
                        task_order='Unknown',
                        program_name=program_name,
                        board_code=board_code,
                        image_name=imgname,
                        ngtype=ngtype,
                        result=res
                    )
                    # 复制图片到展示文件夹
                    copy_to_display_folder(image_path, checkresult)
                    # 按规则复制图片到OK/NG文件夹
                    if collect == 1:
                        if checkresult == "AING":
                            dest_folder = ngPath
                        else:
                            dest_folder = okPath
                        refid = row.get('REFID', '')
                        copy_image_with_suffix(image_path, dest_folder, ngtype, refid)
                    df.at[index, 'FLAG'] = checkresult
                elif ngtype == 'shift':
                    results = model.predict(image_path, conf=okrange)
                    print(f"模型预测结果: {results}")  # 打印模型预测结果
                    checkresult = "AIOK"
                    device_type = "UNK"
                    angle = 0
                    res = "未识别"
                    for result in results:
                        obb = result.obb
                        if len(obb.xywhr) == 0:
                            checkresult = "AING"
                            logging.info(f"图片 {imgname}，未识别，复判结果: NG")
                            continue
                        # 这里 YOLO 返回的是原图坐标，center_x/y 也是原图中心，所以可以直接计算
                        x, y, r = obb.xywhr[0][0], obb.xywhr[0][1], obb.xywhr[0][-1]
                        distance = math.hypot(x - center_x, y - center_y)
                        angle_degrees = math.degrees(r)
                        cls_id = int(obb.cls.item())
                        device_type = class_names[cls_id]
                        angle = round(angle_degrees)
                        logging.info(f"图片 {imgname}，偏移距离: {distance:.2f} pixels, 旋转角度: {angle_degrees:.2f} degrees")
                        if device_type in ['SOP', 'ChipR', 'SOT23', 'QFP']:
                            # 注意：如果分辨率变大，偏移 30 像素的物理意义可能变小，根据实际情况可能需要调整此阈值
                            if distance >= 30 or (30 <= angle_degrees <= 150):
                                checkresult = "AING"
                                res = "NG"
                            else:
                                checkresult = "AIOK"
                                res = "OK"
                        logging.info(f"图片 {imgname}，复判结果: {res}")
                        print(f"图片 {imgname}，偏移距离: {distance:.2f} pixels, 旋转角度: {angle_degrees:.2f} degrees")
                    # 写入历史记录 (非 'fly' 类型)
                    write_to_history(
                        task_order='Unknown',
                        program_name=program_name,
                        board_code=board_code,
                        image_name=imgname,
                        ngtype=ngtype,
                        result=res
                    )
                    # 复制图片到展示文件夹
                    copy_to_display_folder(image_path, checkresult)
                    # 按规则复制图片到OK/NG文件夹（不再创建子文件夹）3
                    if collect == 1:
                        if checkresult == "AING":
                            dest_folder = ngPath
                        else:
                            dest_folder = okPath
                        # 传入ngtype和refid两个参数
                        refid = row.get('REFID', '')        # 缺陷类型
                        copy_image_with_suffix(image_path, dest_folder, ngtype, refid)
                    df.at[index, 'FLAG'] = checkresult
                elif ngtype in ('foreign', 'foregin'):
                    results = modelForeign.predict(image_path, conf=okrange)
                    print(f"模型预测结果: {results}")  # 打印模型预测结果
                    checkresult = "AIOK"
                    device_type = "UNK"
                    res = "OK"
                    
                    ng_found = False  # 标记是否发现NG
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # 获取检测框坐标 (x1, y1, x2, y2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # 判断是否包含图片中心点 (center_x/y 已经是原图中心)
                            contains_center = (x1 <= center_x <= x2) and (y1 <= center_y <= y2)
                            
                            # 如果class_id不为2且检测框包含图片中心点
                            if class_id != 2 and contains_center:
                                checkresult = "AING"
                                res = "NG"
                                ng_found = True
                                logging.info(f"图片 {imgname}，未识别，复判结果: NG")
                                break
                        
                        if ng_found:
                            break
                    
                    # 如果没有发现NG情况，保持默认的OK状态
                    if not ng_found:
                        checkresult = "AIOK"
                        res = "OK"
                        logging.info(f"图片 {imgname}，未识别，复判结果: OK")
                    
                    # 写入历史记录 (非 'fly' 类型)
                    write_to_history(
                        task_order='Unknown',
                        program_name=program_name,
                        board_code=board_code,
                        image_name=imgname,
                        ngtype=ngtype,
                        result=res
                    )
                    
                    # 复制图片到展示文件夹
                    copy_to_display_folder(image_path, checkresult)
                    
                    # 按规则复制图片到OK/NG文件夹
                    if collect == 1:
                        if checkresult == "AING":
                            dest_folder = ngPath
                        else:
                            dest_folder = okPath
                        # 传入ngtype和refid两个参数
                        refid = row.get('REFID', '')        # 缺陷类型
                        copy_image_with_suffix(image_path, dest_folder, ngtype, refid)
                    
                    df.at[index, 'FLAG'] = checkresult

                else:
                    print("未知缺陷类型")
                    df.at[index, 'FLAG'] = "AING"

            df.to_csv(csvPath, index=False)
            shutil.copy(csvPath, resPath)
            shutil.copy(csvPath, historyPath)
            os.remove(csvPath)
            logging.info("复判完成")
            if False:
                try:
                    shutil.rmtree(checkPath)
                    logging.info(f"已删除文件夹: {checkPath}")
                except Exception as e:
                    logging.error(f"删除文件夹失败: {e}")
        time.sleep(2)