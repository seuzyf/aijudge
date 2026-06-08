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
    支持一个窗口下多算法全扫描，分别调用独立分支并精准改写。
    返回当前板复判后是否所有窗口全为 OK (True/False)。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"XML 文件解析失败 {xml_path}: {e}")
        return False

    board_code = root.findtext('.//Barcode') or "Unknown"
    program_name = root.findtext('.//JobName') or "Unknown"
    task_order = root.findtext('.//LotNo') or "Unknown"
    part_count = len(root.findall('.//PartData'))

    logging.info(f"开始复判单板: Board={board_code}, Job={program_name}, Lot={task_order}, 元件数={part_count}")
    modified = False
    total_win_checked = 0
    total_win_ok = 0

    # 外层循环：遍历所有元件
    for idx, part in enumerate(root.findall('.//PartData'), 1):
        parent_id = part.findtext('ParentId')
        part_id = part.findtext('ID')
        
        if not parent_id or not part_id:
            logging.warning(f"跳过元件 {idx}：ParentId 或 ID 为空")
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
                logging.warning(f"图片不存在，跳过元件: 尝试路径 {image_path} 及备用名 {alt_names}")
                continue

        logging.info(f"正在处理元件 {idx}/{part_count}: {image_name}")
        img_for_dim = cv2.imread(image_path)
        if img_for_dim is None:
            logging.warning(f"无法读取图片: {image_path}")
            continue
            
        h_img, w_img = img_for_dim.shape[:2]
        center_x, center_y = w_img / 2.0, h_img / 2.0

        # 内层循环：遍历该元件下的所有判定窗口
        for window in part.findall('.//WindowDataList/WindowData'):
            win_id = window.findtext('ID')
            enable_node = window.find('ENABLE')
            if enable_node is None or enable_node.text != 'True':
                continue
                
            algo_nodes = window.findall('.//AlgorithmDataList/AlgorithmData')
            algo_types = [n.findtext('Type') for n in algo_nodes]
            logging.debug(f"窗口 {win_id} 包含算法类型: {algo_types}")
            
            window_all_ok = True
            has_supported_algo = False
            total_win_checked += 1

            for algo_node in algo_nodes:
                algo_type = algo_node.findtext('Type')
                if algo_type not in ['3', '6']:
                    window_all_ok = False
                    continue
                    
                has_supported_algo = True
                checkresult = "AING"
                res = "NG"
                algo_log_name = f"Type{algo_type}_{win_id}"

                # ================= 算法独立分支路由 =================
                if algo_type == '3':
                    if model is not None:
                        results = model.predict(image_path, conf=okrange, verbose=False)
                        total_obbs = 0
                        for result in results:
                            if result.obb and len(result.obb.xywhr) > 0:
                                total_obbs += len(result.obb.xywhr)
                        logging.info(f"图片 {image_name} 窗口{win_id} Type3 检测到 {total_obbs} 个 OBB")

                        for result in results:
                            if result.obb and len(result.obb.xywhr) > 0:
                                x, y, r = result.obb.xywhr[0][0], result.obb.xywhr[0][1], result.obb.xywhr[0][-1]
                                distance = math.hypot(x - center_x, y - center_y)
                                angle_degrees = math.degrees(r)
                                cls_id = int(result.obb.cls[0].item())
                                device_type = class_names[cls_id] if cls_id < len(class_names) else "UNK"
                                
                                if device_type in ['SOP', 'ChipR', 'SOT23', 'QFP']:
                                    if distance >= 30 or (30 <= angle_degrees <= 150):
                                        checkresult, res = "AING", "NG"
                                    else:
                                        checkresult, res = "AIOK", "OK"
                                        break
                        logging.info(f"图片 {image_name} 窗口{win_id} Type3 判定: {checkresult}")

                elif algo_type == '6':
                    if ocr_model is not None:
                        new_w, new_h = w_img // 3, h_img // 3
                        start_x, start_y = (w_img - new_w) // 2, (h_img - new_h) // 2
                        cropped_image = img_for_dim[start_y:start_y + new_h, start_x:start_x + new_w]
                        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                        enhanced_image = cv2.convertScaleAbs(gray_image, alpha=3, beta=0)
                        
                        temp_image_path = image_path.replace(".jpg", f"_temp_{win_id}_{algo_type}.jpg")
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
                        # 打桩：强制 NG
                        checkresult, res = "AING", "NG"
                        logging.info(f"图片 {image_name} 窗口{win_id} Type6 判定: {checkresult}")

                # 记录本轮算法独立复判结果
                write_to_history(task_order, program_name, board_code, image_name, algo_log_name, res)
                copy_to_display_folder(image_path, checkresult)
                
                if collect == 1:
                    dest_folder = okPath if checkresult == "AIOK" else ngPath
                    copy_image_with_suffix(image_path, dest_folder, algo_log_name, res)

                # 精准修改当前独立算法节点的属性
                if res == "OK":
                    algo_enable = algo_node.find('ENABLE')
                    if algo_enable is not None:
                        algo_enable.text = 'False'
                    logging.info(f"精准修正: 图像 {image_name} 的 {algo_log_name} 算法被判定为误报 (OK)")
                    total_win_ok += 1
                else:
                    window_all_ok = False

            # 只有当该窗口下所有被扫描算法全部判定为 OK 时，才关闭整个窗口
            if window_all_ok and has_supported_algo:
                enable_node.text = 'False'
                modified = True
                logging.info(f"成功修正整个窗口: 图像 {image_name} 的窗口 ID {win_id} 被安全关闭")

    logging.info(f"单板 {board_code} 复判完成：共检查窗口 {total_win_checked} 个，修正为 OK 的窗口 {total_win_ok} 个")

    # 再次检查：判定整块板复判后是否还有残留的真实 NG 窗口
    any_ng_remaining = False
    for w in root.findall('.//WindowDataList/WindowData'):
        if w.findtext('ENABLE') == 'True':
            any_ng_remaining = True
            break

    # 依然有缺陷，则将修改后的结构覆写原 XML
    if modified and any_ng_remaining:
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"单板复判完毕，有残留缺陷，已更新结构化 XML: {xml_path}")
        except Exception as e:
            logging.error(f"覆写保存 XML 失败: {e}")

    if not any_ng_remaining:
        # 全 OK 板，生成 OK 内容并返回 True
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
    """
    整包协同搬运机制：自动根据最内层 Image 路径向外溯源推演，
    同步打包搬运对应的 TempInspResult 与 ResultData 目录，完成移交后彻底删除输入路径下的空源文件夹。
    如果全 OK，自动将缺陷包转换为包含单 Total result OK.xml 文件的标准 OK 板格式（不搬运其他协同目录）。
    """
    rel_path = os.path.relpath(root_dir, src_base)
    paths_to_sync = [rel_path]

    # 根据 Image 的相对路径定位其他数据包的镜像相对路径
    if rel_path.startswith("Image"):
        temp_rel = rel_path.replace("Image", "TempInspResult", 1)
        if os.path.exists(os.path.join(src_base, temp_rel)):
            paths_to_sync.append(temp_rel)
            
        timestamp_dir = os.path.basename(rel_path)
        res_data_rel = os.path.join("ResultData", timestamp_dir)
        if os.path.exists(os.path.join(src_base, res_data_rel)):
            paths_to_sync.append(res_data_rel)

    # 遍历所有协同路径执行原样原结构转移
    for rel in paths_to_sync:
        src_path = os.path.join(src_base, rel)
        target_dir = os.path.join(dst_base, rel)

        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
            except Exception as e:
                logging.error(f"清除维修站旧包冲突失败: {e}")
                continue

        try:
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)

            # 对应转换要求：如果 AI 判定整块板误报转为全 OK，且当前在处理 Image 目录
            if is_now_all_ok and rel.startswith("Image"):
                os.makedirs(target_dir, exist_ok=True)
                ok_xml_file_path = os.path.join(target_dir, "Total result OK.xml")
                
                content = getattr(sync_and_move_board_packages, 'ok_xml_content', '')
                with open(ok_xml_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"AI 成功将当前单板包转为 OK 板格式移交维修站: {target_dir}")
                continue  # 全 OK 时不搬运其他目录
            
            # 正常整包、或者原生 OK 包，执行物理搬运
            if not is_now_all_ok:
                shutil.move(src_path, target_dir)
                logging.info(f"单板协同数据原样移交维修站完成: {target_dir}")
            else:
                logging.info(f"全 OK 板，跳过搬运 {rel} 目录")

        except Exception as e:
            logging.error(f"协同包搬移至维修站失败 {rel}: {e}")

    # 清洗机制：安全移除源路径对应文件夹，并逐层向上传递清除留下的空父目录
    for rel in paths_to_sync:
        src_path = os.path.join(src_base, rel)
        if os.path.exists(src_path):
            try:
                shutil.rmtree(src_path)
                logging.info(f"已清理源路径: {src_path}")
            except Exception as e:
                logging.error(f"清理源路径失败 {src_path}: {e}")

        parent = os.path.dirname(src_path)
        while parent != src_base and os.path.exists(parent):
            if not os.listdir(parent):  # 目录空了就直接干掉
                try:
                    os.rmdir(parent)
                    logging.info(f"源头输入空路径已成功清除: {parent}")
                    parent = os.path.dirname(parent)
                except Exception as e:
                    logging.error(f"删除空目录失败 {parent}: {e}")
                    break
            else:
                break

def process_all_files(check, directory, resPath):
    """
    监控引擎主控逻辑，深度扫描到最内层。
    兼容只配置总目录“奔创AOI测试后输出数据”层，自动分流处理原生 OK 包和复判 NG 包。
    确保 input 目录下所有文件/文件夹最终都同步到 output，无遗漏。
    """
    logging.info(f"奔创SMTv2智能复判深度监控启动。监控源: {directory}, 维修站目标: {resPath}")
    
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

        # 第一步：将根目录下非核心文件夹（如 FiduResult）原样移至 output
        for item in os.listdir(directory):
            if item in ['Image', 'TempInspResult', 'ResultData']:
                continue
            src_item = os.path.join(directory, item)
            dst_item = os.path.join(resPath, item)
            if os.path.exists(src_item):
                try:
                    if os.path.exists(dst_item):
                        if os.path.isdir(dst_item):
                            shutil.rmtree(dst_item)
                        else:
                            os.remove(dst_item)
                    shutil.move(src_item, dst_item)
                    logging.info(f"根目录非核心数据移交: {src_item} -> {dst_item}")
                except Exception as e:
                    logging.error(f"移交根目录条目失败 {src_item}: {e}")

        # 第二步：扫描 Image 树，定位最深层单板目录并进行复判/搬运
        walk_snapshot = list(os.walk(directory))
        
        for root, dirs, files in walk_snapshot:
            if any(skip in root for skip in ['history', 'display', 'AI']):
                continue

            # 分支 1：扫描到原生 OK 板数据包
            if "Total result OK.xml" in files:
                xml_path = os.path.join(root, "Total result OK.xml")
                logging.info(f"扫描到原生 OK 板数据包: {xml_path}，挂起 {lag} 秒等待文件完整写入...")
                time.sleep(lag)
                
                sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=False)
                continue

            # 分支 2：扫描到常规缺陷 NG 数据包，启动核心复判逻辑
            if "Total result NG.xml" in files:
                xml_path = os.path.join(root, "Total result NG.xml")
                logging.info(f"扫描到复判缺陷数据包: {xml_path}，挂起 {lag} 秒以防 IO 延迟...")
                time.sleep(lag)
                
                is_now_all_ok = process_single_xml(xml_path, root, okrange, collect, okPath, ngPath)
                
                sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=is_now_all_ok)

        time.sleep(2)
