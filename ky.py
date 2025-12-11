import cv2
import math
import os
from paddleocr import DocImgOrientationClassification
from ultralytics import YOLO
import time
import pandas as pd
import shutil
import logging
import numpy as np
from datetime import datetime
import csv
import io

# 设置环境变量
os.environ['PADDLE_USE_MKLDNN'] = '0'

# --- 移除全局的 basicConfig，改为在函数内配置，防止被覆盖 ---
# 只保留一个简单的配置防止导入时报错，但实际工作由函数内配置接管
logging.getLogger().setLevel(logging.INFO)

# 加载模型
try:
    model = YOLO("./model/ky.pt")  # 使用 ky.pt
    # 这里的日志可能无法写入文件，因为还没配置好，打印到控制台即可
    print("Loading KY model...")
    model.eval()
    
    ocr_model = DocImgOrientationClassification(model_dir="./model/PP-LCNet_x1_0_doc_ori")
    print("Loading OCR model done.")
except Exception as e:
    print(f"模型加载失败: {e}")

# 全局变量
DISPLAY_FOLDER = "display"
os.makedirs(DISPLAY_FOLDER, exist_ok=True)
disp = 1
processed_folders = set()
IGNORED_FOLDERS = {'history', 'display', 'OK', 'NG', 'temp', 'img', 'model', 'dist', '__pycache__', 'System Volume Information'}

def setup_worker_logger():
    """在子进程中强制配置日志"""
    logger = logging.getLogger()
    #以此清除旧的handler，防止重复打印
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    
    logger.setLevel(logging.INFO)
    
    # 添加文件处理器 (使用gbk编码，与主界面读取一致)
    file_handler = logging.FileHandler('check.log', mode='a', encoding='gbk')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 可选：也添加到控制台，方便调试
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def cv2_imread_safe(file_path):
    """安全读取图片，支持中文路径"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return None

def copy_to_display_folder(image_path, result):
    """将图片按 {result}_{count}.jpg 命名拷贝到 display 文件夹"""
    global disp
    disp += 1
    new_name = f"{result}_{disp}.jpg"
    dest_path = os.path.join(DISPLAY_FOLDER, new_name)
    try:
        shutil.copy(image_path, dest_path)
    except Exception as e:
        logging.error(f"复制展示图片失败: {e}")

def read_threshold_from_config(config_path='config.txt'):
    default_okrange = 0.5
    default_collect = 0
    okrange_found = False
    collect_found = False
    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f"okRange={default_okrange}\n")
                f.write(f"collect={default_collect}\n")
            return default_okrange, default_collect
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("okRange"):
                    _, val = line.strip().split("=")
                    okrange_value = float(val)
                    okrange_found = True
                elif line.startswith("collect"):
                    _, val = line.strip().split("=")
                    collect_value = int(val)
                    collect_found = True
        with open(config_path, 'a') as f:
            if not okrange_found:
                f.write(f"okRange={default_okrange}\n")
            if not collect_found:
                f.write(f"collect={default_collect}\n")
        return okrange_value, collect_value
    except Exception as e:
        logging.error(f"配置读取失败: {e}")
        return default_okrange, default_collect

def copy_image_with_suffix(src_path, dest_folder, ngtype, result):
    """归档图片"""
    try:
        base_name, ext = os.path.splitext(os.path.basename(src_path))
        ngtype_safe = str(ngtype).replace('/', '_').replace('(', '').replace(')', '').replace(' ', '')
        base_filename = f"{base_name}_{ngtype_safe}_{result}{ext}"
        dest_path = os.path.join(dest_folder, base_filename)
        counter = 1
        while os.path.exists(dest_path):
            new_filename = f"{base_name}_{ngtype_safe}_{result}_{counter}{ext}"
            dest_path = os.path.join(dest_folder, new_filename)
            counter += 1
        shutil.copy(src_path, dest_path)
    except Exception as e:
        logging.error(f"归档图片失败: {e}")

def write_to_history(task_order, program_name, board_code, image_name, ngtype, result):
    csv_file = 'history.csv'
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = [task_order, program_name, str(board_code), image_name, ngtype, result, current_date]
    expected_header = ['任务令', '程序名', '单板条码', '图片名', '缺陷类型', '结果', '日期']
    
    try:
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(expected_header)
                writer.writerow(new_row)
            return
        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
    except Exception as e:
        logging.error(f"写入历史记录失败: {e}")

def process_all_files(check, directory, resPath):
    # === 关键修复：进入子进程任务后，立即重新初始化日志 ===
    setup_worker_logger()
    
    logging.info("KY设备处理模式已启动，扫描路径：%s", directory)
    
    # 路径处理
    if not resPath or not os.path.exists(resPath):
        if resPath:
            try:
                os.makedirs(resPath, exist_ok=True)
                historyPath = os.path.join(resPath, 'history')
            except:
                historyPath = os.path.join(os.path.dirname(directory), 'history')
        else:
            historyPath = os.path.join(os.path.dirname(directory), 'history')
    else:
        historyPath = os.path.join(resPath, 'history')
        
    okPath = os.path.join(historyPath, 'OK')
    ngPath = os.path.join(historyPath, 'NG')
    os.makedirs(historyPath, exist_ok=True)
    os.makedirs(okPath, exist_ok=True)
    os.makedirs(ngPath, exist_ok=True)
    
    global IGNORED_FOLDERS
    IGNORED_FOLDERS.add(os.path.basename(historyPath))

    global disp
    disp = 1
    global processed_folders

    while check:
        okrange, collect = read_threshold_from_config()
        
        if not os.path.exists(directory):
            time.sleep(2)
            continue
            
        items = os.listdir(directory)
        
        # 过滤掉系统生成的文件夹
        folders = []
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and item not in IGNORED_FOLDERS:
                folders.append(item_path)
        
        current_folders_set = set(folders)
        processed_folders = processed_folders.intersection(current_folders_set)

        target_folder = None
        for folder in folders:
            if folder not in processed_folders:
                target_folder = folder
                break
        
        if not target_folder:
            time.sleep(1)
            continue

        logging.info(f"发现新文件夹: {target_folder}")
        
        csv_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.csv')]
        if not csv_files:
            logging.warning(f"文件夹 {target_folder} 中未找到CSV文件，跳过。")
            processed_folders.add(target_folder)
            continue
            
        csv_file_path = os.path.join(target_folder, csv_files[0])
        
        try:
            # 智能读取 CSV
            try:
                df = pd.read_csv(csv_file_path, sep=None, engine='python', encoding='utf-8-sig')
            except Exception:
                try:
                    df = pd.read_csv(csv_file_path, sep=None, engine='python', encoding='gbk')
                except Exception as e:
                    logging.error(f"CSV读取失败: {e}")
                    processed_folders.add(target_folder)
                    continue
            
            df.columns = df.columns.str.strip()
            
            required_columns = ['Judgment item/defect type', 'Image Name', 'Inspection results']
            if not all(col in df.columns for col in required_columns):
                logging.error(f"CSV列名不匹配。需要: {required_columns}, 实际: {df.columns.tolist()}")
                processed_folders.add(target_folder)
                continue

            for index, row in df.iterrows():
                ng_full_str = str(row.get('Judgment item/defect type', ''))
                image_name = str(row.get('Image Name', '')).strip()
                
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                mission_order = str(row.get('Mission Order', 'Unknown'))
                program_name = str(row.get('Program Name', 'Unknown'))
                barcode = str(row.get('Barcode', 'Unknown'))
                
                image_path = os.path.join(target_folder, image_name)
                
                if not os.path.exists(image_path):
                    logging.warning(f"图片文件丢失: {image_path}")
                    continue

                img_data = cv2_imread_safe(image_path)
                if img_data is None:
                    logging.error(f"无法读取图片 (可能是文件损坏): {image_path}")
                    continue
                
                h, w = img_data.shape[:2]
                center_x, center_y = w / 2.0, h / 2.0
                
                res = "NG"
                checkresult = "AING"
                
                # 判定逻辑
                if "(1700)OCROCV" in ng_full_str:
                    logging.info(f"正在处理 OCR 图片: {image_name}")
                    try:
                        # 解析目标角度
                        file_stem = os.path.splitext(image_name)[0]
                        parts = file_stem.split('_')
                        target_angle = None
                        
                        # 尝试解析最后一部分是否为数字
                        if len(parts) > 1:
                            try:
                                target_angle = float(parts[-1])
                            except ValueError:
                                # 如果最后一部分不是数字（比如 J1103），说明文件名格式不对
                                pass
                        
                        if target_angle is not None:
                            # --- 只有解析到角度才进行模型推理 ---
                            new_w, new_h = w // 3, h // 3
                            start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
                            cropped_image = img_data[start_y:start_y + new_h, start_x:start_x + new_w]
                            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                            enhanced_image = cv2.convertScaleAbs(gray_image, alpha=3, beta=0)
                            
                            temp_ocr_path = os.path.join(target_folder, "temp_ocr_check.jpg")
                            # 保存临时文件
                            cv2.imencode('.jpg', enhanced_image)[1].tofile(temp_ocr_path)
                            
                            ocr_output = ocr_model.predict(temp_ocr_path, batch_size=1)
                            if os.path.exists(temp_ocr_path):
                                os.remove(temp_ocr_path)
                                
                            ocr_pred_angle = None
                            if ocr_output and len(ocr_output) > 0 and isinstance(ocr_output[0], dict):
                                result_dict = ocr_output[0]
                                label_names = result_dict.get('label_names', [])
                                scores = result_dict.get('scores', [])
                                if label_names and scores:
                                    if float(scores[0]) >= okrange:
                                        ocr_pred_angle = float(label_names[0])
                            
                            # 比较角度 (考虑 0, 90, 180, 270)
                            valid_angles = [0.0, 90.0, 180.0, 270.0]
                            if ocr_pred_angle is not None and ocr_pred_angle in valid_angles and target_angle in valid_angles:
                                if ocr_pred_angle == target_angle:
                                    res = "OK"
                                    checkresult = "AIOK"
                            
                            logging.info(f"OCR复判: 目标={target_angle}, 识别={ocr_pred_angle}, 复判结果: {res}")
                        
                        else:
                            # === 新增：如果无法解析角度，打印警告 ===
                            logging.warning(f"OCR复判跳过: 文件名 '{image_name}' 无法提取角度 (需如 1_J1101_90.jpg), 默认结果: {res}")

                    except Exception as e:
                        logging.error(f"OCR处理出错: {e}")

                elif "(1100)Overhang" in ng_full_str:
                    logging.info(f"正在处理 Overhang 图片: {image_name}")
                    try:
                        # 抑制 YOLO 控制台输出（verbose=False），防止刷屏，主要依靠我们的 log
                        results = model.predict(image_path, conf=okrange, verbose=False)
                        is_centered = False
                        
                        if results:
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                                            is_centered = True
                                            break
                                
                                elif result.obb is not None:
                                    for poly in result.obb.xyxyxyxy:
                                        pts = poly.cpu().numpy().astype(int)
                                        dist = cv2.pointPolygonTest(pts, (center_x, center_y), False)
                                        if dist >= 0:
                                            is_centered = True
                                            break
                                
                                if is_centered:
                                    break
                        
                        if is_centered:
                            res = "OK"
                            checkresult = "AIOK"
                        else:
                            res = "NG"
                            checkresult = "AING"
                        
                        logging.info(f"Overhang复判: 框在中心={is_centered}, 复判结果: {res}")
                    except Exception as e:
                        logging.error(f"Overhang处理出错: {e}", exc_info=True)

                else:
                    res = "NG"
                    checkresult = "AING"
                    logging.info(f"未定义类型: {ng_full_str}, 复判结果: {res}")

                df.at[index, 'Inspection results'] = checkresult
                write_to_history(mission_order, program_name, barcode, image_name, ng_full_str, res)
                copy_to_display_folder(image_path, checkresult)
                
                if collect == 1:
                    dest_folder = okPath if checkresult == "AIOK" else ngPath
                    copy_image_with_suffix(image_path, dest_folder, ng_full_str, barcode)

            # 保存并覆盖
            df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
            logging.info(f"CSV文件已更新: {csv_file_path}")
            
            try:
                shutil.copy(csv_file_path, historyPath)
            except: pass

            processed_folders.add(target_folder)
            
        except Exception as e:
            logging.error(f"处理文件夹出错 {target_folder}: {e}")
            processed_folders.add(target_folder)

        time.sleep(1)