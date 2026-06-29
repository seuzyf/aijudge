import cv2
import math
import os
import time
import shutil
import logging
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np
import onnxruntime as ort

os.environ['PADDLE_USE_MKLDNN'] = '0'

# ================= 定义类别名称列表 =================
class_names = ['ChipR', 'SOP', 'SOT23', 'QFP']

# ================= 载入算法模型 (全 ONNX 架构) =================
yolo_session = None
ocr_session = None

# 使用 CPU 引擎，完美兼容老旧工控机
providers = ['CPUExecutionProvider']

try:
    yolo_session = ort.InferenceSession("./model/benchuang.onnx", providers=providers)
    logging.info("载入模型：奔创偏位检测模型 (ONNX) 成功")
except Exception as e:
    logging.error(f"偏位模型加载失败: {e}")
    print(f"偏位模型加载提示: {e}")

try:
    ocr_session = ort.InferenceSession("./model/PP-LCNet_doc_ori.onnx", providers=providers)
    logging.info("载入模型：奔创OCR方向模型 (ONNX) 成功")
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

def update_resultdata_xml(xml_path, ok_window_ids):
    """
    针对 ResultData.xml 的专用修复逻辑
    将指定的窗口 <sWindId> 匹配，并将其 <m_bOk> 置为 1
    :param ok_window_ids: 集合，包含需要置 OK 的窗口 ID (字符串形式的数字，如 {'1','3'})
    """
    if not os.path.exists(xml_path) or not ok_window_ids:
        return False
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False
        
        for insp_param in root.iter('InspParamTemp_Defect'):
            # 优先用 sWindId 匹配
            s_wind_id_node = insp_param.find('sWindId')
            if s_wind_id_node is not None and s_wind_id_node.text:
                wind_id = s_wind_id_node.text.strip()
                if wind_id in ok_window_ids:
                    m_bok_node = insp_param.find('m_bOk')
                    if m_bok_node is not None and m_bok_node.text != '1':
                        m_bok_node.text = '1'
                        modified = True
                        logging.info(f"  --> [ResultData同步] 窗口 sWindId={wind_id} 命中复判OK, <m_bOk> 已强制置为 1")
            else:
                # 如果没有 sWindId，回退尝试用 wndName 匹配（保持兼容）
                wnd_name_node = insp_param.find('wndName')
                if wnd_name_node is not None and wnd_name_node.text:
                    name = wnd_name_node.text.strip()
                    if name.startswith('window'):
                        maybe_id = name[6:]  # 提取数字部分
                        if maybe_id in ok_window_ids:
                            m_bok_node = insp_param.find('m_bOk')
                            if m_bok_node is not None and m_bok_node.text != '1':
                                m_bok_node.text = '1'
                                modified = True
                                logging.info(f"  --> [ResultData同步] 窗口 wndName={name} 命中复判OK (回退匹配), <m_bOk> 已置为 1")
                        
        if modified:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"  --> [ResultData同步] 成功保存更新: {xml_path}")
        else:
            logging.warning(f"  --> [ResultData同步] 未在 {xml_path} 中找到匹配的窗口ID {ok_window_ids}，未做修改")
        return True
    except Exception as e:
        logging.error(f"处理 ResultData.xml 时发生异常: {e}")
        return False


def process_single_xml(xml_path, folder_path, okrange, collect, okPath, ngPath):
    """核心：使用 ElementTree 结构化解析 Total result NG.xml。"""
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
    
    # ======== 收集复判为 OK 的 Type6 窗口的数字 ID ========
    type6_ok_window_ids = set()

    for idx, node in enumerate(loop_nodes, 1):
        if is_container_mode:
            part = node.find('.//PartData') or node.find('.//PARTDATA')
            if part is None: continue
        else:
            part = node
            
        parent_id = get_text_ignore_case(part, ['ParentId', 'PARENTID', 'ParentID'])
        part_id = get_text_ignore_case(part, ['ID', 'Id', 'id'])
        package_name = get_text_ignore_case(part, ['PackageName', 'PACKAGENAME', 'Packagename'])
        
        if parent_id == "Unknown" or part_id == "Unknown":
            continue

        # ================= 暴力深度遍历获取角度 =================
        xml_angle = None
        for child in part.iter():
            tag_name = child.tag.split('}')[-1].lower()
            if tag_name == 'roi':
                for sub in child.iter():
                    sub_tag = sub.tag.split('}')[-1].lower()
                    if sub_tag == 'a' and sub.text is not None:
                        try:
                            xml_angle = float(sub.text.strip())
                            logging.info(f"  --> [XML] 元件 {parent_id}@{part_id} 成功提取原始角度: {xml_angle}°")
                        except ValueError:
                            pass
                break
        # =======================================================
            
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
        
        # ================== 核心修复：绕过 OpenCV 的中文路径读取 Bug ==================
        try:
            img_for_dim = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            img_for_dim = None

        if img_for_dim is None:
            logging.warning(f"无法读取图片: {image_path}")
            continue
        # ==============================================================================
            
        h_img, w_img = img_for_dim.shape[:2]
        center_x, center_y = w_img / 2.0, h_img / 2.0

        window_nodes = node.findall('.//WindowData') or node.findall('.//WINDOWDATA') or node.findall('.//windowdata')
        
        for window in window_nodes:
            # ---------- 提取窗口ID并处理成纯数字 ----------
            raw_win_id = get_text_ignore_case(window, ['ID', 'Id', 'id'])
            # 如果ID形如 "window1"，提取数字部分；否则直接使用
            if raw_win_id.lower().startswith('window'):
                win_id = raw_win_id[6:]  # 去掉 "window" 前缀，例如 "1"
            else:
                win_id = raw_win_id
            # ------------------------------------------------

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
                    if yolo_session is not None:
                        try:
                            # 动态读取模型输入尺寸 (OBB 通常为 1024x1024)
                            yolo_input_shape = yolo_session.get_inputs()[0].shape
                            model_h = yolo_input_shape[2] if isinstance(yolo_input_shape[2], int) else 1024
                            model_w = yolo_input_shape[3] if isinstance(yolo_input_shape[3], int) else 1024
                            
                            # Letterbox 尺寸缩放
                            scale = min(model_w / w_img, model_h / h_img)
                            new_unpad_w, new_unpad_h = int(round(w_img * scale)), int(round(h_img * scale))
                            dw, dh = (model_w - new_unpad_w) / 2, (model_h - new_unpad_h) / 2
                            
                            resized = cv2.resize(img_for_dim, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
                            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                            
                            blob = padded[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                            blob = np.expand_dims(blob, axis=0)
                            
                            yolo_out = yolo_session.run(None, {yolo_session.get_inputs()[0].name: blob})[0]
                            preds = yolo_out[0].T

                            class_scores = preds[:, 4:-1]
                            max_scores = np.max(class_scores, axis=1)
                            best_idx = np.argmax(max_scores)
                            best_score = max_scores[best_idx]

                            if best_score < (okrange + 0.5):
                                logging.info(f"  --> [Type3] 窗口:{win_id} 未检测到满足阈值的目标框，判定: NG")
                            else:
                                best_pred = preds[best_idx]
                                cls_id = int(np.argmax(best_pred[4:-1]))
                                pred_x, pred_y = best_pred[0], best_pred[1]
                                pred_r = best_pred[-1]
                                
                                orig_x = (pred_x - left) / scale
                                orig_y = (pred_y - top) / scale
                                
                                distance = math.hypot(orig_x - center_x, orig_y - center_y)
                                angle_degrees = math.degrees(pred_r)
                                device_type = class_names[cls_id] if cls_id < len(class_names) else "UNK"

                                temp_res = "NG"
                                if device_type in ['SOP', 'ChipR', 'SOT23', 'QFP']:
                                    if distance >= 30 or (30 <= angle_degrees <= 150):
                                        temp_res = "NG"
                                    else:
                                        temp_res = "OK"

                                logging.info(f"  --> [Type3] 类别:{device_type} 置信度:{best_score:.3f} 角度:{angle_degrees:.2f}° 偏移距:{distance:.2f} 判定: {temp_res}")

                                if temp_res == "OK":
                                    checkresult, res = "AIOK", "OK"
                                    
                        except Exception as e:
                            logging.error(f"  --> [Type3] 处理 OBB 检测时发生异常: {e}", exc_info=True)
                    else:
                        logging.info(f"  --> [Type3] 进入模型未加载异常分支，强制锁定结果为 NG")
                                
                    logging.info(f" -> 窗口 {win_id} Type3 最终判定: {checkresult}")

                elif algo_type == '6':
                    checkresult, res = "AING", "NG"
                    if ocr_session is not None:
                        try:
                            logging.info(f"  --> [Type6] 当前元件封装类型: {package_name}")
                            if 'RF' in package_name.upper():
                                roi_ratio = 0.6
                                alpha_val = 1.5
                                beta_val = -50
                            else:
                                roi_ratio = 0.6
                                alpha_val = 3.0
                                beta_val = 10
                            logging.info(f"  --> [Type6] 应用预处理参数 (ROI: {roi_ratio*100}%, Alpha: {alpha_val}, Beta: {beta_val})")

                            new_w = int(w_img * roi_ratio)
                            new_h = int(h_img * roi_ratio)
                            start_x = (w_img - new_w) // 2
                            start_y = (h_img - new_h) // 2
                            cropped_image = img_for_dim[start_y:start_y + new_h, start_x:start_x + new_w]
                            
                            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                            enhanced_image = cv2.convertScaleAbs(gray_image, alpha=alpha_val, beta=beta_val)

                            img_resized = cv2.resize(enhanced_image, (224, 224))
                            if len(img_resized.shape) == 2:
                                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                            else:
                                img_resized = cv2.cvtColor(img_resized, cv2.BGR2RGB)
                                
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            img_normalized -= np.array([0.485, 0.456, 0.406])
                            img_normalized /= np.array([0.229, 0.224, 0.225])
                            
                            input_tensor = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)

                            ocr_out = ocr_session.run(None, {ocr_session.get_inputs()[0].name: input_tensor})[0]
                            
                            probs = np.exp(ocr_out[0]) / np.sum(np.exp(ocr_out[0]))
                            class_id = int(np.argmax(probs))
                            ocr_score = float(probs[class_id])
                            
                            angles_map = [0.0, 90.0, 180.0, 270.0]
                            ocr_angle = None
                            
                            if class_id < len(angles_map):
                                ocr_angle = angles_map[class_id]
                                logging.info(f"  --> [Type6] OCR预测角度: {ocr_angle}°, 置信度: {ocr_score:.3f}")
                                
                                if ocr_score < okrange:
                                    ocr_angle = None
                                    logging.warning("  --> [Type6] 进入OCR置信度低于阈值分支，忽略预测角度，判定NG")
                            else:
                                logging.info(f"  --> [Type6] OCR无预测角度")

                            if ocr_angle is not None:
                                if xml_angle is not None:
                                    cad_to_image_angle = {
                                        0.0: 0.0, 45.0: 0.0, -315.0: 0.0, 60.0: 0.0, -300.0: 0.0, 300.0: 0.0, -60.0: 0.0,
                                        90.0: 270.0, -270.0: 270.0, 135.0: 270.0, -225.0: 270.0, 150.0: 270.0, -210.0: 270.0, 30.0: 270.0, -150.0: 270.0,
                                        180.0: 180.0, -180.0: 180.0, 225.0: 180.0, -135.0: 180.0, 240.0: 180.0, -120.0: 180.0, 120.0: 180.0, -240.0: 180.0,
                                        270.0: 90.0, -90.0: 90.0, 315.0: 90.0, -45.0: 90.0, 330.0: 90.0, -30.0: 90.0, 210.0: 90.0, -330.0: 90.0
                                    }
                                    
                                    if xml_angle in cad_to_image_angle:
                                        expected_ocr_angle = cad_to_image_angle[xml_angle]
                                        if ocr_angle == expected_ocr_angle:
                                            res = "OK"
                                            checkresult = "AIOK"
                                            logging.info(f"  --> [Type6] OCR角度({ocr_angle}°) 与 CAD等效角度({expected_ocr_angle}° <- {xml_angle}°) 一致。复判结果: OK")
                                            # ========= 核心联动：记录 Type6 成功的窗口数字 ID =========
                                            type6_ok_window_ids.add(win_id)
                                            # ========================================================
                                        else:
                                            logging.info(f"  --> [Type6] OCR角度({ocr_angle}°) 与 CAD等效角度({expected_ocr_angle}° <- {xml_angle}°) 不一致。复判结果: NG")
                                    else:
                                        logging.info(f"  --> [Type6] CAD角度 {xml_angle}° 未在等效映射表中。复判结果: NG")
                                else:
                                    logging.info(f"  --> [Type6] 当前元件在XML中无有效角度值，保持判NG")
                                    
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

    # ================= 联动执行 ResultData.xml 的修复动作 =================
    if type6_ok_window_ids:
        try:
            parts = Path(folder_path).parts
            if 'Image' in parts:
                idx = parts.index('Image')
                new_parts = list(parts)
                new_parts[idx] = 'TempInspResult'
                resultdata_dir = Path(*new_parts)
                resultdata_xml_path = resultdata_dir / "ResultData.xml"
                
                if resultdata_xml_path.exists():
                    update_resultdata_xml(str(resultdata_xml_path), type6_ok_window_ids)
                else:
                    logging.warning(f"  --> [ResultData同步] 未找到对应的关联文件: {resultdata_xml_path}")
        except Exception as e:
            logging.error(f"  --> [ResultData同步] 自动寻址与修改发生异常: {e}")
    # ====================================================================

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

def remove_empty_dirs(path, stop_path):
    """
    向上递归清理空层级目录，避免复判完成后留下一堆层级空文件夹
    直到遇到基础安全目录 (stop_path) 为止
    """
    try:
        path = os.path.abspath(path)
        stop_path = os.path.abspath(stop_path)
        current = os.path.dirname(path)
        
        while current and current.startswith(stop_path) and current != stop_path:
            if os.path.exists(current) and os.path.isdir(current) and not os.listdir(current):
                os.rmdir(current)
                logging.info(f"已自动向上清理空单板层级目录: {current}")
                current = os.path.dirname(current)
            else:
                break
    except Exception as e:
        logging.error(f"清理单板空目录机制报错: {e}")

def sync_and_move_board_packages(root_dir, src_base, dst_base, is_now_all_ok=False):
    """
    整包协同转移机制：
    1. 动态生成包含 Image / TempInspResult / ResultData / FiduResult 等核心全套数据结构。
    2. 执行物理转移：将数据同步（复制）到维修站路径。
    3. 清理源文件时，【绝对限制】只允许清理 Image 下的判过单板文件夹，其他所有附随文件夹必须被保留在原位。
    """
    rel_path = os.path.relpath(root_dir, src_base)
    paths_to_sync = [rel_path]
    parts = Path(rel_path).parts

    # 动态构建单板级别的相关附随路径
    if len(parts) > 0 and parts[0] == "Image":
        # 追加 TempInspResult 层
        temp_rel = rel_path.replace("Image", "TempInspResult", 1)
        if os.path.exists(os.path.join(src_base, temp_rel)):
            paths_to_sync.append(temp_rel)
            
        # 追加 ResultData 层
        timestamp_dir = parts[-1]
        res_data_rel = os.path.join("ResultData", timestamp_dir)
        if os.path.exists(os.path.join(src_base, res_data_rel)):
            paths_to_sync.append(res_data_rel)
            
        # ⭐ 追加 FiduResult 层 (格式为：FiduResult/Group@Board@Board)
        if len(parts) >= 3:
            group_name = parts[1]
            board_name = parts[2]
            fidu_rel = os.path.join("FiduResult", f"{group_name}@{board_name}@{board_name}")
            if os.path.exists(os.path.join(src_base, fidu_rel)) and fidu_rel not in paths_to_sync:
                paths_to_sync.append(fidu_rel)

    for rel in paths_to_sync:
        src_path = os.path.join(src_base, rel)
        target_dir = os.path.join(dst_base, rel)

        if not os.path.exists(src_path):
            continue

        # 仅当路径属于 Image 文件夹时才赋予物理删除权限
        is_image_path = rel.startswith("Image")

        try:
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)

            # ======== 1. 同步 / 复制操作（写往维修站）========
            if is_now_all_ok and is_image_path:
                # 判为OK且是图片路径，只在维修站生成 OK xml，不全部复刻大量图片过去
                os.makedirs(target_dir, exist_ok=True)
                ok_xml_file_path = os.path.join(target_dir, "Total result OK.xml")
                
                content = getattr(sync_and_move_board_packages, 'ok_xml_content', '')
                with open(ok_xml_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"AI 成功将当前单板包转为 OK 板格式写入维修站: {target_dir}")
            else:
                # 其他情况 (NG的Image, 或所有 TempInspResult/ResultData/FiduResult 等非Image路径) 均安全复制
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
                else:
                    shutil.copy(src_path, target_dir)
                logging.info(f"协同数据复制同步至维修站: {target_dir}")

            # ======== 2. 删除 / 清理操作（仅对源 Image 执行）========
            if is_image_path:
                if os.path.isdir(src_path):
                    shutil.rmtree(src_path)
                else:
                    os.remove(src_path)
                logging.info(f"已删除源判图触发数据: {src_path}")
                
                # 安全停止路径仅保留到 Image/HUAWEI 层
                stop_path = src_base
                rel_parts = Path(rel).parts
                if len(rel_parts) >= 3:
                    stop_path = os.path.join(src_base, rel_parts[0], rel_parts[1])
                
                remove_empty_dirs(src_path, stop_path)

        except Exception as e:
            logging.error(f"协同包转移至维修站或源文件清理过程中失败 {rel}: {e}")

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
    
    # 跟踪已处理的路径，以防意外
    processed_items = set()

    while check:

        processed_items = {p for p in processed_items if os.path.exists(p)}
        
        # 记录本轮真实处理过的数据板时间戳(目录名通常即为时间戳)
        processed_timestamps = set()
        
        okrange, collect, lag = read_threshold_from_config()
        
        if not os.path.exists(directory):
            time.sleep(3)
            logging.info(f"监控源为空: {directory}")
            continue

        processed_any_file = False

        # ---------------- 步骤 1：处理浅层外围及游离文件 ----------------
        try:
            for item in os.listdir(directory):
                src_item = os.path.join(directory, item)
                if item == "NGBufferDataList.csv":
                    continue # NGBufferDataList.csv 最后处理
                
                # 处理根目录非核心文件夹 (保持原样，纯复制不删源)
                if item not in ['Image', 'TempInspResult', 'ResultData', 'FiduResult'] and os.path.isdir(src_item):
                    if src_item not in processed_items:
                        dst_item = os.path.join(resPath, item)
                        try:
                            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                            logging.info(f"根目录非核心数据正常复制: {src_item} -> {dst_item}")
                            processed_any_file = True
                            processed_items.add(src_item)
                        except Exception as e:
                            logging.error(f"复制根目录条目失败 {src_item}: {e}")

                # 处理核心文件夹内直接存在的游离文件 (保持原样，纯复制不删源)
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
            
            # 直接在当前层级的 dirs 中移除不需要遍历的系统/结果文件夹。
            dirs[:] = [d for d in dirs if d not in ['history', 'display', 'AI', 'InspectResult']]

            lower_files = [f.lower() for f in files]

            if "total result ok.xml" in lower_files:
                exact_name = next(f for f in files if f.lower() == "total result ok.xml")
                xml_path = os.path.join(root, exact_name)
                if xml_path not in processed_items:
                    logging.info(f"扫描到原生 OK 板数据包，挂起 {lag} 秒等待文件完整写入...")
                    time.sleep(lag)
                    sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=False)
                    processed_items.add(xml_path)
                    
                    # 采集处理完毕的单板时间戳作为 CSV 清理的信标
                    timestamp_dir = os.path.basename(root)
                    processed_timestamps.add(timestamp_dir)
                    processed_any_file = True
                continue

            if "total result ng.xml" in lower_files:
                exact_name = next(f for f in files if f.lower() == "total result ng.xml")
                xml_path = os.path.join(root, exact_name)
                if xml_path not in processed_items:
                    logging.info(f"扫描到复判缺陷数据包，挂起 {lag} 秒以防 IO 延迟...")
                    time.sleep(lag)
                    
                    is_now_all_ok = process_single_xml(xml_path, root, okrange, collect, okPath, ngPath)
                    
                    sync_and_move_board_packages(root, directory, resPath, is_now_all_ok=is_now_all_ok)
                    processed_items.add(xml_path)
                    
                    # 采集处理完毕的单板时间戳作为 CSV 清理的信标
                    timestamp_dir = os.path.basename(root)
                    processed_timestamps.add(timestamp_dir)
                    processed_any_file = True

        # ---------------- 步骤 3：NGBufferDataList.csv 精确流转与重排逻辑 ----------------
        # 仅当此轮实际完成了板子复判（信标被点亮）时才介入修改 CSV
        if processed_timestamps:
            buffer_src = os.path.join(directory, 'TempInspResult', 'NGBufferDataList.csv')
            buffer_dst = os.path.join(resPath, 'TempInspResult', 'NGBufferDataList.csv')
            
            if os.path.exists(buffer_src):
                try:
                    os.makedirs(os.path.dirname(buffer_dst), exist_ok=True)
                    
                    src_keep_lines = []
                    src_move_lines = []
                    header = "Key, Board path"
                    
                    # 1. 拆包源 CSV：通过时间戳精确剥离被处理过的板子
                    with open(buffer_src, 'r', encoding='utf-8-sig', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            header = lines[0].strip()
                            for line in lines[1:]:
                                line = line.strip()
                                if not line: continue
                                
                                # 解析逗号后的 Board path
                                parts = line.split(',', 1)
                                if len(parts) == 2:
                                    bp = parts[1].strip()
                                    # 提取 @ 符号前的时间戳进行匹配比对
                                    timestamp = bp.split('@')[0] 
                                    
                                    if timestamp in processed_timestamps:
                                        src_move_lines.append(bp)
                                    else:
                                        src_keep_lines.append(bp)
                                else:
                                    src_keep_lines.append(line)

                    # 2. 如果成功提取到了本次流转的记录
                    if src_move_lines:
                        # 覆写源路径 CSV：抛弃已处理的，保留未处理的，从 0 重新编号
                        with open(buffer_src, 'w', encoding='utf-8-sig') as f:
                            f.write(header + "\n")
                            for i, bp in enumerate(src_keep_lines):
                                f.write(f"{i}, {bp}\n")
                                
                        # 解析维修站（目标）CSV，预备接收新数据
                        dst_lines = []
                        if os.path.exists(buffer_dst):
                            with open(buffer_dst, 'r', encoding='utf-8-sig', errors='ignore') as f:
                                dst_lines_raw = f.readlines()
                                if len(dst_lines_raw) > 1:
                                    for line in dst_lines_raw[1:]:
                                        line = line.strip()
                                        if not line: continue
                                        parts = line.split(',', 1)
                                        if len(parts) == 2:
                                            dst_lines.append(parts[1].strip())
                                        else:
                                            dst_lines.append(line)
                                            
                        # 将被剥离的板子去重后挂载到维修站的末尾
                        existing_bps = set(dst_lines)
                        for bp in src_move_lines:
                            if bp not in existing_bps:
                                dst_lines.append(bp)
                                
                        # 覆写维修站 CSV：全部重新编号从 0 排列
                        with open(buffer_dst, 'w', encoding='utf-8-sig') as f:
                            f.write(header + "\n")
                            for i, bp in enumerate(dst_lines):
                                f.write(f"{i}, {bp}\n")
                                
                        logging.info(f"NGBufferDataList.csv 增量流转完毕: 从源移除 {len(src_move_lines)} 条单板数据, 保留 {len(src_keep_lines)} 条待处理, 并已安全同步至维修站。")
                    else:
                        logging.info("本次扫描处理未在 CSV 中检索到对应记录，略过更新。")
                        
                except Exception as e:
                    logging.error(f"处理 NGBufferDataList.csv 增量同步时失败: {e}")

        # ---------------- 步骤 4：动态休眠机制 ----------------
        if processed_any_file:
            time.sleep(0.5)
        else:
            time.sleep(3) 
