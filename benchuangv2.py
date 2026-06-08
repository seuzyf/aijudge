import cv2
import math
import os
import time
import shutil
import logging
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

os.environ['PADDLE_USE_MKLDNN'] = '0'

# ================= 定义类别名称列表 (与 benchuangSMT 同步) =================
class_names = ['ChipR', 'SOP', 'SOT23', 'QFP']

# ================= 载入算法模型 =================
model = None
ocr_model = None

try:
    from ultralytics import YOLO
    model = YOLO("./model/benchuang.pt")
    model.eval()
    logging.info("载入模型：奔创偏位检测模型成功")
except Exception as e:
    print(f"模型加载提示: {e}")

try:
    from paddleocr import DocImgOrientationClassification
    ocr_model = DocImgOrientationClassification(model_dir="./model/PP-LCNet_x1_0_doc_ori")
    logging.info("载入模型：奔创OCR检测模型成功")
except Exception as e:
    print(f"OCR模型加载提示: {e}")

logging.basicConfig(
    filename='check.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  
)

DISPLAY_FOLDER = "display"
os.makedirs(DISPLAY_FOLDER, exist_ok=True)
disp = 1

def copy_to_display_folder(image_path, result):
    """将图片按 {result}_{count}.jpg 命名拷贝到 display 文件夹，供前端界面展示"""
    global disp
    disp += 1
    new_name = f"{result}_{disp}.jpg"
    dest_path = os.path.join(DISPLAY_FOLDER, new_name)
    try:
        shutil.copy(image_path, dest_path)
    except Exception as e:
        pass

def read_threshold_from_config(config_path='config.txt'):
    default_okrange = 0.5
    default_collect = 0
    default_lag = 1.0  
    
    okrange_value = default_okrange
    collect_value = default_collect
    lag_value = default_lag

    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f"okRange={default_okrange}\ncollect={default_collect}\nlag={default_lag}\n")
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
                elif line.startswith("collect"):
                    _, val = line.split("=")
                    collect_value = int(val)
                elif line.startswith("lag"):
                    _, val = line.split("=")
                    lag_value = float(val)
                    
        return okrange_value, collect_value, lag_value
    except Exception as e:
        logging.error(f"配置读取失败: {e}")
        return default_okrange, default_collect, default_lag

def copy_image_with_suffix(src_path, dest_folder, ngtype, result):
    """缺陷图片收集函数"""
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    ngtype_safe = str(ngtype).replace('/', '_')
    base_filename = f"{base_name}_{ngtype_safe}_{result}{ext}"
    dest_path = os.path.join(dest_folder, base_filename)
    counter = 1
    while os.path.exists(dest_path):
        new_filename = f"{base_name}_{ngtype_safe}_{result}_{counter}{ext}"
        dest_path = os.path.join(dest_folder, new_filename)
        counter += 1
    shutil.copy(src_path, dest_path)

def write_to_history(task_order, program_name, board_code, image_name, ngtype, result):
    """将复判历史记录写入 CSV"""
    csv_file = 'history.csv'
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = [task_order, program_name, str(board_code), image_name, ngtype, result, current_date]
    expected_header = ['任务令', '程序名', '单板条码', '图片名', '缺陷类型', '结果', '日期']
    
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)
            writer.writerow(new_row)
        return
        
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

def process_single_xml(xml_path, folder_path, okrange, collect, okPath, ngPath):
    """
    核心：使用 ElementTree 结构化解析 Total result NG.xml。
    多算法共用图片，分别复判，独立修改 XML 节点。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"XML 文件解析失败 {xml_path}: {e}")
        return

    board_code = root.findtext('.//Barcode') or "Unknown"
    program_name = root.findtext('.//JobName') or "Unknown"
    task_order = root.findtext('.//LotNo') or "Unknown"

    modified = False

    # 外层循环：遍历所有元件 (一个 PartData 对应一张实拍图片)
    for part in root.findall('.//PartData'):
        parent_id = part.findtext('ParentId')
        part_id = part.findtext('ID')
        
        if not parent_id or not part_id:
            continue
            
        # 组装FOV图片名称，格式如：21@186_AC.jpg
        image_name = f"{parent_id}@{part_id}_AC.jpg"
        image_path = os.path.join(folder_path, image_name)

        # 图片探测容错机制
        if not os.path.exists(image_path):
            alt_names = [f"{parent_id}@{part_id}.jpg", f"{part_id}_AC.jpg", f"{part_id}.jpg"]
            for alt in alt_names:
                if os.path.exists(os.path.join(folder_path, alt)):
                    image_path = os.path.join(folder_path, alt)
                    image_name = alt
                    break
            if not os.path.exists(image_path):
                logging.warning(f"图片缺失，跳过此元件: {image_name}")
                continue

        # 预加载图片到内存（避免内层多个算法重复读取硬盘）
        img_for_dim = cv2.imread(image_path)
        if img_for_dim is None:
            logging.error(f"OpenCV 读取图像损坏: {image_path}")
            continue
            
        h_img, w_img = img_for_dim.shape[:2]
        center_x, center_y = w_img / 2.0, h_img / 2.0

        # 内层循环：遍历该元件下的所有判定窗口 (不同算法)
        for window in part.findall('.//WindowDataList/WindowData'):
            win_id = window.findtext('ID')
            
            # 定位该窗口的NG标识节点 (一般为 <ENABLE>True</ENABLE>)
            enable_node = window.find('ENABLE')
            if enable_node is None or enable_node.text != 'True':
                continue # 如果不是 NG (True)，则跳过复判
                
            checkresult = "AING"
            res = "NG"

            # 取窗口内的第一个算法节点作为判据
            algo_node = window.find('.//AlgorithmDataList/AlgorithmData')
            algo_type = algo_node.findtext('Type') if algo_node is not None else "Unknown"
            algo_log_name = f"Type{algo_type}_{win_id}"

            # ================= 算法路由分支 =================
            if algo_type == '3':
                # 分支 1：Type 为 3 (对应 benchuangSMT 的 shift)
                if model is not None:
                    results = model.predict(image_path, conf=okrange, verbose=False)
                    for result in results:
                        if result.obb and len(result.obb.xywhr) > 0:
                            x, y, r = result.obb.xywhr[0][0], result.obb.xywhr[0][1], result.obb.xywhr[0][-1]
                            distance = math.hypot(x - center_x, y - center_y)
                            angle_degrees = math.degrees(r)
                            cls_id = int(result.obb.cls[0].item())
                            device_type = class_names[cls_id] if cls_id < len(class_names) else "UNK"
                            
                            logging.info(f"图片 {image_name}，偏移距离: {distance:.2f} pixels, 旋转角度: {angle_degrees:.2f} degrees")
                            
                            if device_type in ['SOP', 'ChipR', 'SOT23', 'QFP']:
                                if distance >= 30 or (30 <= angle_degrees <= 150):
                                    checkresult, res = "AING", "NG"
                                else:
                                    checkresult, res = "AIOK", "OK"
                                    break # 确认找到OK，跳出结果遍历
                            else:
                                # 非这四个类型的，如果不做要求默认维持原判
                                pass

            elif algo_type == '6':
                # 分支 2：Type 为 6 (对应 OCR 角度识别)
                if ocr_model is not None:
                    new_w, new_h = w_img // 3, h_img // 3
                    start_x, start_y = (w_img - new_w) // 2, (h_img - new_h) // 2
                    
                    # 裁剪中心区域并灰度强化
                    cropped_image = img_for_dim[start_y:start_y + new_h, start_x:start_x + new_w]
                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=3, beta=0)
                    
                    # 生成供 PaddleOCR 读取的临时图
                    temp_image_path = image_path.replace(".jpg", f"_temp_{win_id}.jpg")
                    cv2.imwrite(temp_image_path, enhanced_image)
                    
                    ocr_output = ocr_model.predict(temp_image_path, batch_size=1)
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                    
                    if ocr_output and len(ocr_output) > 0 and isinstance(ocr_output[0], dict):
                        scores = ocr_output[0].get('scores', [])
                        label_names = ocr_output[0].get('label_names', [])
                        if scores and label_names:
                            ocr_angle = float(label_names[0])
                            ocr_score = float(scores[0])
                            logging.info(f"图片 {image_name} OCR预测角度: {ocr_angle}°, 置信度: {ocr_score}")
                            
                # == 打桩测试点 ==：不论预测角度为多少，强制将其判定为 NG
                checkresult, res = "AING", "NG"

            else:
                # 别的不判，全部维持 NG 判定
                logging.info(f"图片 {image_name} 的未定义算法 Type={algo_type} 强制判定为 NG")
                pass
            
            # 记录本轮复判结果
            write_to_history(task_order, program_name, board_code, image_name, algo_log_name, res)
            copy_to_display_folder(image_path, checkresult)
            
            if collect == 1:
                dest_folder = okPath if checkresult == "AIOK" else ngPath
                copy_image_with_suffix(image_path, dest_folder, algo_log_name, res)

            # 只有当 AI 判定该【特定算法】误报时，才精准修改这一个节点的属性
            if res == "OK":
                enable_node.text = 'False'
                modified = True
                logging.info(f"成功修正: 图像 {image_name} 的 {algo_log_name} 算法被 AI 判定为误报 (OK)")

    # 遍历结束，如有修改则将整个树结构无损覆写回 XML 文件
    if modified:
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"单板复判完毕，已安全覆写更新结构化 XML: {xml_path}")
        except Exception as e:
            logging.error(f"覆写保存 XML 失败: {e}")

def process_all_files(check, directory, resPath):
    """
    监控引擎主控逻辑，包含层级遍历与结果搬运（同 Saki 直接移交维修站）
    """
    logging.info(f"奔创SMTv2智能复判启动。读取监控源: {directory}, 写入维修站目标: {resPath}")
    
    parent_dir = os.path.dirname(directory) if os.path.dirname(directory) else '.'
    historyPath = os.path.join(parent_dir, 'history')
    okPath = os.path.join(historyPath, 'OK')
    ngPath = os.path.join(historyPath, 'NG')
    os.makedirs(okPath, exist_ok=True)
    os.makedirs(ngPath, exist_ok=True)
    
    global disp
    disp = 1

    while check:
        okrange, collect, lag = read_threshold_from_config()
        
        if not os.path.exists(directory):
            time.sleep(2)
            continue

        # 深度探测所有子层级
        for root, dirs, files in os.walk(directory):
            if any(skip in root for skip in ['history', 'display', 'AI']):
                continue

            if "Total result NG.xml" in files:
                xml_path = os.path.join(root, "Total result NG.xml")
                
                logging.info(f"定位到数据包: {xml_path}，为防止图片未写入完毕挂起 {lag} 秒...")
                time.sleep(lag)
                
                # 开始解析复判该文件夹下的 XML，并覆写原本 XML
                process_single_xml(xml_path, root, okrange, collect, okPath, ngPath)
                
                # 提取相对路径层级，整包搬运 (类 Saki 直接抛去维修站目录)
                rel_path = os.path.relpath(root, directory)
                target_dir = os.path.join(resPath, rel_path)
                
                if os.path.exists(target_dir):
                    try:
                        shutil.rmtree(target_dir)
                    except Exception as e:
                        logging.error(f"清除目标旧目录失败: {e}")
                        continue
                        
                try:
                    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
                    shutil.move(root, target_dir)
                    logging.info(f"单板任务移交维修站完成: {target_dir}")
                except Exception as e:
                    logging.error(f"文件树搬移失败: {e}")
        
        time.sleep(2)
