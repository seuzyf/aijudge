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

# ================= 定义类别名称列表 =================
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
    logging.error(f"偏位模型加载失败: {e}")
    print(f"偏位模型加载提示: {e}")

try:
    from paddleocr import DocImgOrientationClassification
    ocr_model = DocImgOrientationClassification(model_dir="./model/PP-LCNet_x1_0_doc_ori")
    logging.info("载入模型：奔创OCR检测模型成功")
except Exception as e:
    logging.error(f"OCR模型加载失败: {e}")
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

# ================= 动态获取XML标准角度 =================
def get_angle_from_xml(xml_path):
    """
    从子XML获取元件标准角度。查找 <PartData> 下的 <Roi> <a> 值。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        part_data = root.find('.//PartData') or root.find('.//PARTDATA')
        if part_data is not None:
            roi_elem = part_data.find('.//Roi') or part_data.find('.//ROI')
            if roi_elem is not None:
                angle_elem = roi_elem.find('a') or roi_elem.find('A')
                if angle_elem is not None and angle_elem.text is not None:
                    try:
                        angle_value = float(angle_elem.text)
                        # 特殊角度转换对齐
                        if angle_value == 270.0:
                            angle_value = 90.0
                        elif angle_value == 90.0:
                            angle_value = 270.0
                        logging.info(f"成功从XML获取标准角度并转换后: {angle_value}°")
                        return angle_value
                    except ValueError:
                        logging.warning(f"XML文件 {xml_path} 中的 angle 值无法转换为浮点数。")
                        return None
                else:
                    logging.info("进入 XML 未找到 <a> 标签分支")
            else:
                logging.info("进入 XML 未找到 <Roi> 标签分支")
        else:
            logging.info("进入 XML 未找到 <PartData> 标签分支")
    except Exception as e:
        logging.error(f"解析XML文件 {xml_path} 失败: {e}")
    return None

def copy_to_display_folder(image_path, result):
    """将图片按 {result}{count}.jpg 命名拷贝到 display 文件夹，供前端界面展示"""
    global disp
    disp += 1
    new_name = f"{result}{disp}.jpg"
    dest_path = os.path.join(DISPLAY_FOLDER, new_name)
    try:
        shutil.copy(image_path, dest_path)
    except Exception as e:
        pass

# 全局变量缓存配置
_cached_config = (0.5, 0, 1.0)
_last_config_mtime = 0

def read_threshold_from_config(config_path='config.txt'):
    global _cached_config, _last_config_mtime
    default_okrange, default_collect, default_lag = 0.5, 0, 1.0

    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f"okRange={default_okrange}\ncollect={default_collect}\nlag={default_lag}\n")
            return default_okrange, default_collect, default_lag
            
        current_mtime = os.path.getmtime(config_path)
        if current_mtime == _last_config_mtime:
            return _cached_config

        with open(config_path, 'r') as f:
            okrange_value, collect_value, lag_value = default_okrange, default_collect, default_lag
            for line in f:
                line = line.strip()
                if not line or '=' not in line: continue
                if line.startswith("okRange"): okrange_value = float(line.split("=")[1])
                elif line.startswith("collect"): collect_value = int(line.split("=")[1])
                elif line.startswith("lag"): lag_value = float(line.split("=")[1])
        
        _cached_config = (okrange_value, collect_value, lag_value)
        _last_config_mtime = current_mtime
        logging.info(f"配置已更新: okRange={okrange_value}, collect={collect_value}, lag={lag_value}")
        
        return _cached_config
    except Exception as e:
        logging.error(f"配置读取失败: {e}")
        return default_okrange, default_collect, default_lag

def copy_image_with_suffix(src_path, dest_folder, ngtype, result):
    """缺陷图片收集函数"""
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    ngtype_safe = str(ngtype).replace('/', '')
    base_filename = f"{base_name}{ngtype_safe}{result}{ext}"
    dest_path = os.path.join(dest_folder, base_filename)
    counter = 1
    while os.path.exists(dest_path):
        new_filename = f"{base_name}{ngtype_safe}{result}{counter}{ext}"
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
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"XML 文件解析失败 {xml_path}: {e}")
        return False

    def get_text_ignore_case(node, tags):
        if node is None: return "Unknown"
        for tag in tags:
            val = node.findtext(f'.//{tag}')
            if val: return val
        return "Unknown"

    board_code = get_text_ignore_case(root, ['Barcode', 'BARCODE', 'barcode'])
    program_name = get_text_ignore_case(root, ['JobName', 'JOBNAME', 'Jobname'])
    task_order = get_text_ignore_case(root, ['LotNo', 'LOTNO', 'Lotno'])
    
    container_nodes = root.findall('.//RawDataContainer') or root.findall('.//RAWDATACONTAINER')
    
    if container_nodes:
        loop_nodes = container_nodes
        is_container_mode = True
    else:
        loop_nodes = root.findall('.//PartData') or root.findall('.//PARTDATA')
        is_container_mode = False

    part_count = len(loop_nodes)
    logging.info(f"开始复判单板: Board={board_code}, Job={program_name}, Lot={task_order}, 元件数={part_count}")
    
    modified = False
    total_win_checked = 0
    total_win_ok = 0

    for idx, node in enumerate(loop_nodes, 1):
        if is_container_mode:
            part = node.find('.//PartData') or node.find('.//PARTDATA')
            if part is None: continue
        else:
            part = node
            
        parent_id = get_text_ignore_case(part, ['ParentId', 'PARENTID', 'ParentID'])
        part_id = get_text_ignore_case(part, ['ID', 'Id', 'id'])
        
        if parent_id == "Unknown" or part_id == "Unknown":
            continue
            
        image_name = f"{parent_id}@{part_id}_AC.jpg"
        image_path = os.path.join(folder_path, image_name)

        if not os.path.exists(image_path):
            alt_names = [f"{parent_id}@{part_id}.jpg", f"{part_id}_AC.jpg", f"{part_id}.jpg"]
            found = False
            for alt in alt_names:
                alt_path = os.path.join(folder_path, alt)
                if os.path.exists(alt_path):
                    image_path = alt_path
                    image_name = alt
                    found = True
                    break
            if not found:
                logging.warning(f"图片不存在，跳过元件: 尝试备用名均失败 {alt_names}")
                continue

        logging.info(f"正在处理元件 {idx}/{part_count}: {image_name}")
        img_for_dim = cv2.imread(image_path)
        if img_for_dim is None:
            logging.warning(f"无法读取图片: {image_path}")
            continue
            
        h_img, w_img = img_for_dim.shape[:2]
        center_x, center_y = w_img / 2.0, h_img / 2.0

        window_nodes = node.findall('.//WindowData') or node.findall('.//WINDOWDATA') or node.findall('.//windowdata')
        
        for window in window_nodes:
            win_id = get_text_ignore_case(window, ['ID', 'Id', 'id'])
            
            enable_node = None
            for e_tag in ['ENABLE', 'Enable', 'enable', 'IsEnable']:
                enable_node = window.find(e_tag)
                if enable_node is not None: break
                
            if enable_node is None or not enable_node.text:
                continue
                
            if enable_node.text.strip().lower() != 'true':
                continue
                
            algo_nodes = window.findall('.//AlgorithmData') or window.findall('.//ALGORITHMDATA') or window.findall('.//algorithmdata')
            
            window_all_ok = True
            has_supported_algo = False
            total_win_checked += 1

            for algo_node in algo_nodes:
                algo_type = get_text_ignore_case(algo_node, ['Type', 'TYPE', 'type'])
                if algo_type not in ['3', '6']:
                    logging.info(f"进入 algo_type 不支持跳过分支，当前类型: {algo_type}")
                    window_all_ok = False
                    continue
                    
                has_supported_algo = True
                checkresult = "AING"
                res = "NG"
                algo_log_name = f"Type{algo_type}_{win_id}"

                if algo_type == '3':
                    if model is not None:
                        results = model.predict(image_path, conf=okrange, verbose=False)
                        total_obbs = sum(len(r.obb.xywhr) for r in results if r.obb)
                        
                        if total_obbs == 0:
                            logging.info(f"  --> [Type3] 窗口:{win_id} 未检测到目标框，判定: NG")
                        
                        for result in results:
                            if result.obb and len(result.obb.xywhr) > 0:
                                x, y, r = result.obb.xywhr[0][0], result.obb.xywhr[0][1], result.obb.xywhr[0][-1]
                                conf = float(result.obb.conf[0].item())
                                distance = math.hypot(x - center_x, y - center_y)
                                angle_degrees = math.degrees(r)
                                cls_id = int(result.obb.cls[0].item())
                                device_type = class_names[cls_id] if cls_id < len(class_names) else "UNK"
                                
                                temp_res = "NG"
                                if device_type in ['SOP', 'ChipR', 'SOT23', 'QFP']:
                                    if distance >= 30 or (30 <= angle_degrees <= 150):
                                        temp_res = "NG"
                                    else:
                                        temp_res = "OK"
                                
                                logging.info(f"  --> [Type3] 类别:{device_type} 置信度:{conf:.3f} 角度:{angle_degrees:.2f}° 偏移距:{distance:.2f} 判定: {temp_res}")
                                
                                if temp_res == "OK":
                                    checkresult, res = "AIOK", "OK"
                                    break
                    else:
                        logging.info(f"  --> [Type3] 进入模型未加载异常分支，强制锁定结果为 NG")
                                
                    logging.info(f" -> 窗口 {win_id} Type3 最终判定: {checkresult}")

                elif algo_type == '6':
                    checkresult, res = "AING", "NG"
                    
                    if ocr_model is not None:
                        try:
                            # 1. 裁剪画面正中心的1/3区域
                            new_w = w_img // 3
                            new_h = h_img // 3
                            start_x = (w_img - new_w) // 2
                            start_y = (h_img - new_h) // 2
                            cropped_image = img_for_dim[start_y:start_y + new_h, start_x:start_x + new_w]
                            
                            # 2. 转换为灰度图并增强对比度/亮度
                            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                            enhanced_image = cv2.convertScaleAbs(gray_image, alpha=3, beta=0)

                            # 3. 将处理后的图像保存为临时路径供模型预测使用
                            temp_image_path = image_path.replace(".jpg", f"_enhanced_tmp_{win_id}.jpg")
                            cv2.imwrite(temp_image_path, enhanced_image)

                            # 4. 调用 OCR 模型进行预测
                            ocr_output = ocr_model.predict(temp_image_path, batch_size=1)
                            ocr_angle = None
                            ocr_score = None
                            
                            # 删除临时图片
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                                
                            if ocr_output and len(ocr_output) > 0:
                                first_result = ocr_output[0]
                                if isinstance(first_result, dict):
                                    label_names = first_result.get('label_names', [])
                                    scores = first_result.get('scores', [])

                                    if label_names and len(label_names) > 0 and scores and len(scores) > 0:
                                        ocr_angle_str = label_names[0]
                                        ocr_angle = float(ocr_angle_str)
                                        ocr_score = float(scores[0])
                                        logging.info(f"  --> [Type6] OCR预测角度: {ocr_angle}°, 置信度: {ocr_score:.3f}")
                                        
                                        if ocr_score < okrange:
                                            ocr_angle = None
                                            logging.warning("  --> [Type6] 进入OCR置信度低于阈值分支，忽略预测角度，判定NG")

                            if ocr_angle is not None:
                                # 构造对应的XML文件路径以获取标准角度
                                xml_filename = os.path.splitext(os.path.basename(image_path))[0] + ".xml"
                                xml_path_for_angle = os.path.join(folder_path, xml_filename)
                                
                                xml_angle = get_angle_from_xml(xml_path_for_angle)
                                
                                if xml_angle is not None:
                                    valid_angles = [0.0, 90.0, 180.0, 270.0]
                                    if ocr_angle in valid_angles and xml_angle in valid_angles:
                                        if ocr_angle == xml_angle:
                                            res = "OK"
                                            checkresult = "AIOK"
                                            logging.info(f"  --> [Type6] OCR角度({ocr_angle}°) 与 XML标准角度({xml_angle}°) 一致。复判结果: OK")
                                        else:
                                            logging.info(f"  --> [Type6] OCR角度({ocr_angle}°) 与 XML标准角度({xml_angle}°) 不一致。复判结果: NG")
                                    else:
                                        logging.info(f"  --> [Type6] 进入无效角度分支，标准或预测不在0/90/180/270中。复判结果: NG")
                                else:
                                    logging.info(f"  --> [Type6] 进入获取XML标准角度失败分支，保持判NG")
                                    
                        except Exception as e:
                            logging.error(f"  --> [Type6] 处理 'OCR' 时发生异常: {e}", exc_info=True)
                    else:
                        logging.info(f"  --> [Type6] 进入OCR模型未加载异常分支，强制锁定结果为 NG")
                            
                    logging.info(f" -> 窗口 {win_id} Type6 最终判定: {checkresult}")

                # 执行统一的结果保存
                write_to_history(task_order, program_name, board_code, image_name, algo_log_name, res)
                copy_to_display_folder(image_path, checkresult)
                
                if collect == 1:
                    dest_folder = okPath if checkresult == "AIOK" else ngPath
                    copy_image_with_suffix(image_path, dest_folder, algo_log_name, res)

                if res == "OK":
                    algo_enable = None
                    for e_tag in ['ENABLE', 'Enable', 'enable', 'IsEnable']:
                        algo_enable = algo_node.find(e_tag)
                        if algo_enable is not None: break
                    if algo_enable is not None:
                        algo_enable.text = 'False'
                    logging.info(f" -> 精准修正: {algo_log_name} 误报修改为 OK")
                    total_win_ok += 1
                else:
                    window_all_ok = False

            if window_all_ok and has_supported_algo:
                enable_node.text = 'False'
                modified = True
                logging.info(f" -> 成功修正: 整个窗口 ID {win_id} 已安全关闭")

    logging.info(f"单板 {board_code} 复判完成：共检查支持的NG窗口 {total_win_checked} 个，修正为 OK 的窗口 {total_win_ok} 个")

    any_ng_remaining = False
    all_windows = root.findall('.//WindowData') or root.findall('.//WINDOWDATA') or root.findall('.//windowdata')
    for w in all_windows:
        e_node = None
        for e_tag in ['ENABLE', 'Enable', 'enable', 'IsEnable']:
            e_node = w.find(e_tag)
            if e_node is not None: break
        if e_node is not None and e_node.text and e_node.text.strip().lower() == 'true':
            any_ng_remaining = True
            break

    if modified and any_ng_remaining:
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"单板复判完毕，有残留缺陷，已更新结构化 XML: {xml_path}")
        except Exception as e:
            logging.error(f"覆写保存 XML 失败: {e}")

    if not any_ng_remaining:
        inspect_time = datetime.now().strftime('%Y%m%d%H%M%S')
        ok_content = (
            f"<total>\n"
            f"Board Part Count  Part : {part_count};PCB SIZE : X 265 Y 240;Group : HUAWEI;"
            f"Job name : {program_name};Lot no : {task_order};Inspection Time : {inspect_time};"
            f"machine name : ;machine result : OK;"
            f"Board Full image path : \\\\172.16.145.219\\WholeImageB\\SMT27\\Post-AOI00460\\{task_order}\\{program_name.replace(' / ', '_')}_{board_code}_{inspect_time}.jpg;"
            f"Barcode Result : {board_code};\n"
            f"</total>"
        )
        sync_and_move_board_packages.ok_xml_content = ok_content
        return True

    return False

def sync_and_move_board_packages(root_dir, src_base, dst_base, is_now_all_ok=False):
    """整包协同搬运机制：采用复制 (copytree) 不删除原文件。"""
    rel_path = os.path.relpath(root_dir, src_base)
    paths_to_sync = [rel_path]

    if rel_path.startswith("Image"):
        temp_rel = rel_path.replace("Image", "TempInspResult", 1)
        if os.path.exists(os.path.join(src_base, temp_rel)):
            paths_to_sync.append(temp_rel)
            
        timestamp_dir = os.path.basename(rel_path)
        res_data_rel = os.path.join("ResultData", timestamp_dir)
        if os.path.exists(os.path.join(src_base, res_data_rel)):
            paths_to_sync.append(res_data_rel)

    for rel in paths_to_sync:
        src_path = os.path.join(src_base, rel)
        target_dir = os.path.join(dst_base, rel)

        try:
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)

            if is_now_all_ok and rel.startswith("Image"):
                os.makedirs(target_dir, exist_ok=True)
                ok_xml_file_path = os.path.join(target_dir, "Total result OK.xml")
                
                content = getattr(sync_and_move_board_packages, 'ok_xml_content', '')
                with open(ok_xml_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"AI 成功将当前单板包转为 OK 板格式写入维修站: {target_dir}")
                continue
            
            if not is_now_all_ok:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
                else:
                    shutil.copy(src_path, target_dir)
                logging.info(f"单板协同数据原样复制至维修站完成: {target_dir}")
            else:
                logging.info(f"进入全OK跳过数据原样复制分支，略过: {rel} 目录")

        except Exception as e:
            logging.error(f"协同包复制至维修站失败 {rel}: {e}")

def process_all_files(check, directory, resPath):
    """监控引擎主控逻辑，深度扫描到最内层。"""
    logging.info(f"奔创SMTv2智能复判深度监控启动。监控源: {directory}, 维修站目标: {resPath}")

    parent_dir = os.path.dirname(directory) if os.path.dirname(directory) else '.'
    historyPath = os.path.join(parent_dir, 'history')
    okPath = os.path.join(historyPath, 'OK')
    ngPath = os.path.join(historyPath, 'NG')
    os.makedirs(okPath, exist_ok=True)
    os.makedirs(ngPath, exist_ok=True)

    global disp
    disp = 1
    
    # 跟踪已处理的路径，因为改用复制不删原文件，防止进入无限死循环
    processed_items = set()

    while check:
        okrange, collect, lag = read_threshold_from_config()
        
        if not os.path.exists(directory):
            time.sleep(3)
            continue

        processed_any_file = False

        # ---------------- 步骤 1：处理浅层外围及游离文件 ----------------
        try:
            for item in os.listdir(directory):
                src_item = os.path.join(directory, item)
                
                if item == "NGBufferDataList.csv":
                    continue # NGBufferDataList.csv 最后处理
                
                # 处理根目录非核心文件夹 (改为复制)
                if item not in ['Image', 'TempInspResult', 'ResultData', 'FiduResult'] and os.path.isdir(src_item):
                    if src_item not in processed_items:
                        dst_item = os.path.join(resPath, item)
                        try:
                            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                            logging.info(f"根目录非核心数据复制: {src_item} -> {dst_item}")
                            processed_any_file = True
                            processed_items.add(src_item)
                        except Exception as e:
                            logging.error(f"复制根目录条目失败 {src_item}: {e}")

                # 处理核心文件夹内直接存在的游离文件
                if item in ['TempInspResult', 'ResultData', 'Image', 'FiduResult'] and os.path.isdir(src_item):
                    for sub_item in os.listdir(src_item):
                        if sub_item == "NGBufferDataList.csv":
                            continue
                        item_path = os.path.join(src_item, sub_item)
                        if os.path.isfile(item_path):
                            if item_path not in processed_items:
                                dst_dir = os.path.join(resPath, item)
                                os.makedirs(dst_dir, exist_ok=True)
                                dst_path = os.path.join(dst_dir, sub_item)
                                try:
                                    shutil.copy(item_path, dst_path)
                                    logging.info(f"同步全局独立数据文件: {item_path} -> {dst_path}")
                                    processed_any_file = True
                                    processed_items.add(item_path)
                                except Exception as e:
                                    logging.error(f"复制全局独立文件失败 {item_path}: {e}")
        except Exception as e:
            logging.error(f"处理外围目录游离文件时发生错误: {e}")

        # ---------------- 步骤 2：深度遍历核心逻辑 ----------------
        for root, dirs, files in os.walk(directory):
            if any(skip in root for skip in ['history', 'display', 'AI']):
                dirs[:] = [] 
                continue

            if "Total result OK.xml" in files:
                xml_path = os.path.join(root, "Total result OK.xml")
                if xml_path not in processed_items:
                    logging.info(f"扫描到原生 OK 板数据包，挂起 {lag} 秒等待文件完整写入...")
                    time.sleep(lag)
                    sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=False)
                    processed_items.add(xml_path)
                    processed_any_file = True
                continue

            if "Total result NG.xml" in files:
                xml_path = os.path.join(root, "Total result NG.xml")
                if xml_path not in processed_items:
                    logging.info(f"扫描到复判缺陷数据包，挂起 {lag} 秒以防 IO 延迟...")
                    time.sleep(lag)
                    
                    is_now_all_ok = process_single_xml(xml_path, root, okrange, collect, okPath, ngPath)
                    
                    sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=is_now_all_ok)
                    processed_items.add(xml_path)
                    processed_any_file = True

        # ---------------- 步骤 3：最后复制 NGBufferDataList.csv ----------------
        if processed_any_file:
            buffer_src = os.path.join(directory, 'TempInspResult', 'NGBufferDataList.csv')
            buffer_dst = os.path.join(resPath, 'TempInspResult', 'NGBufferDataList.csv')
            if os.path.exists(buffer_src):
                try:
                    os.makedirs(os.path.dirname(buffer_dst), exist_ok=True)
                    shutil.copy(buffer_src, buffer_dst)
                    logging.info(f"成功将 NGBufferDataList.csv 最终复制到: {buffer_dst}")
                except Exception as e:
                    logging.error(f"复制 NGBufferDataList.csv 失败: {e}")

        # ---------------- 步骤 4：动态休眠机制 ----------------
        if processed_any_file:
            time.sleep(0.5)
        else:
            time.sleep(3)
