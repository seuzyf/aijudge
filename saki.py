import re
from datetime import datetime
import os
import glob
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
import pyautogui
import csv

logging.basicConfig(filename='check.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义类别名称列表
class_names = [
    'SR',   # 类别 0
    'SC',   # 类别 1
    'BALL',  # 类别 2
    'BSR',  # 类别 3
    'BSC',   # 类别 4
    'HOLE'
]

# 加载训练好的模型
model = YOLO("./model/saki.pt")  # 替换为你的 best.pt 文件路径

# 👇 新增 display 文件夹配置
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

def find_and_click_icon(icon_path, confidence=0.8):
    # 记录当前鼠标位置
    original_position = pyautogui.position()
    print(f"🖱️ 当前鼠标位置: {original_position}")

    # 截图屏幕
    screenshot = pyautogui.screenshot()
    # 转换为 OpenCV 可处理的格式 (RGB -> BGR)
    screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 读取图标图片
    icon_img = cv2.imread(icon_path)
    if icon_img is None:
        print(f"❌ 无法加载图标文件: {icon_path}")
        return False

    # 使用模板匹配
    result = cv2.matchTemplate(screen_img, icon_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 判断是否匹配成功
    if max_val >= confidence:
        icon_height, icon_width = icon_img.shape[:2]

        # 计算中心点位置
        center_x = max_loc[0] + icon_width // 2
        center_y = max_loc[1] + icon_height // 2

        print(f"✅ 图标位于屏幕坐标: ({center_x}, {center_y}), 置信度: {max_val:.2f}")

        # 移动鼠标并点击
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()

        # 回到原始位置
        pyautogui.moveTo(original_position)
        print(f"🔁 鼠标已移回原位置: {original_position}")

        return True
    else:
        print(f"❌ 未找到相似图标 (最大置信度: {max_val:.2f})")
        return False

def read_threshold_from_config(config_path='config.txt'):
    """
    从 config.txt 文件中读取 OKrange 和 collect 配置。
    如果文件不存在或未指定字段，则创建文件并写入默认值。
    """
    default_okrange = 0.5
    default_collect = 0  # 默认关闭图片收集功能
    okrange_found = False
    collect_found = False

    try:
        # 检查文件是否存在
        if not os.path.exists(config_path):
            logging.info(f"配置文件 {config_path} 不存在，正在创建并写入默认值 okRange={default_okrange}, collect={default_collect}")
            with open(config_path, 'w') as file:
                file.write(f"okRange={default_okrange}\n")
                file.write(f"collect={default_collect}\n")
            return default_okrange, default_collect

        # 读取文件内容
        okrange_value = default_okrange
        collect_value = default_collect

        with open(config_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("okRange"):
                    _, value = line.strip().split("=")
                    okrange_value = float(value)
                    okrange_found = True
                elif line.startswith("collect"):
                    _, value = line.strip().split("=")
                    collect_value = int(value)
                    collect_found = True

        # 如果文件中未找到字段，则追加默认值
        with open(config_path, 'a') as file:
            if not okrange_found:
                logging.info(f"配置文件 {config_path} 中未找到 okRange 字段，正在追加默认值 okRange={default_okrange}")
                file.write(f"okRange={default_okrange}\n")
            if not collect_found:
                logging.info(f"配置文件 {config_path} 中未找到 collect 字段，正在追加默认值 collect={default_collect}")
                file.write(f"collect={default_collect}\n")

        return okrange_value, collect_value

    except Exception as e:
        logging.error(f"读取或更新配置文件失败: {e}，默认使用阈值 okRange={default_okrange}, collect={default_collect}")
        return default_okrange, default_collect

def copy_image_with_suffix(src_path, dest_folder, result):
    """
    复制图片到目标文件夹，避免重名问题。
    将分类结果和概率值添加到文件名中，格式为：原文件名_结果_概率后缀.jpg
    如果目标文件夹中已存在同名图片，则自动添加编号后缀。
    
    :param src_path: 源图片路径
    :param dest_folder: 目标文件夹路径
    :param result: 分类结果（OK/NG）
    :param probability: 分类概率值（0-1之间的浮点数）
    """
    
    # 分离文件名和扩展名
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    
    # 构建基础文件名（包含结果和概率）
    base_filename = f"{base_name}_{result}{ext}"
    dest_path = os.path.join(dest_folder, base_filename)
    
    # 处理文件名冲突
    counter = 1
    while os.path.exists(dest_path):
        # 在概率后添加数字后缀
        new_filename = f"{base_name}_{result}_{counter}{ext}"
        dest_path = os.path.join(dest_folder, new_filename)
        counter += 1

    # 执行复制操作
    shutil.copy(src_path, dest_path)
    logging.info(f"图片已复制到 {dest_path}")

def write_to_history(task_order, program_name, board_code, image_name, ngtype, result):
    csv_file = 'history.csv'
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_row = [
        task_order,
        program_name,
        str(board_code),   # ⬅️ 强制转换为字符串
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

def template_match(input_image_path, temp_folder, threshold):
    """
    使用 temp 文件夹中的模板图片对输入图片进行匹配。
    如果匹配概率最高的框包含输入图片的中心点，并且匹配概率超过阈值，则返回 "OK"；否则返回 "NG"。
    
    :param input_image_path: 输入图片路径
    :param temp_folder: 模板图片文件夹路径
    :param threshold: 匹配概率阈值，默认为 0.5
    :return: "OK" 或 "NG"
    """
    # 检查 temp_folder 是否存在
    if not os.path.exists(temp_folder):
        try:
            os.makedirs(temp_folder)
            logging.info(f"已创建缺失的模板文件夹: {temp_folder}")
        except Exception as e:
            logging.error(f"创建模板文件夹失败: {temp_folder}, 错误: {str(e)}")
        return "NG"
    # 读取输入图片并转换为灰度图
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Unable to read input image at {input_image_path}")
        return "NG"

    # 转换输入图片为灰度图
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 获取输入图片的尺寸和中心点坐标
    ih, iw = input_gray.shape[:2]
    center_x, center_y = iw // 2, ih // 2

    # 初始化变量记录最佳匹配结果
    best_match_probability = -1
    best_match_box = None
    best_template_name = None

    # 遍历 temp 文件夹中的所有模板图片
    for template_name in os.listdir(temp_folder):
        template_path = os.path.join(temp_folder, template_name)
        template = cv2.imread(template_path)
        if template is None:
            print(f"Warning: Unable to read template image at {template_path}. Skipping...")
            continue

        # 转换模板图片为灰度图
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 获取模板图片的尺寸
        th, tw = template_gray.shape[:2]

        # 确保模板图片小于输入图片
        if th > ih or tw > iw:
            print(f"Warning: Template {template_name} is larger than input image. Skipping...")
            continue

        # 使用模板匹配方法进行匹配
        result = cv2.matchTemplate(input_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 获取匹配结果的最大值及其位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc  # 匹配区域的左上角坐标
        bottom_right = (top_left[0] + tw, top_left[1] + th)  # 匹配区域的右下角坐标

        # 打印模板图片的匹配概率
        print(f"Template: {template_name}, Match Probability: {max_val:.4f}")

        # 更新最佳匹配结果（仅考虑超过阈值的匹配）
        if max_val > best_match_probability and max_val >= threshold:
            best_match_probability = max_val
            best_match_box = (top_left, bottom_right)
            best_template_name = template_name

    # 判断最佳匹配框是否包含输入图片的中心点
    if best_match_box is not None:
        top_left, bottom_right = best_match_box
        x1, y1 = top_left
        x2, y2 = bottom_right

        # 检查中心点是否在最佳匹配框内
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            logging.info(f"Best Match: {best_template_name}, Probability: {best_match_probability:.4f}, Match ok")
            return "OK"

    # 如果没有找到包含中心点的匹配框，返回 "NG"
    print("No valid match found.")
    return "NG"

def predict(image_path):
    """
    使用 YOLO 模型进行预测，返回 "OK" 或 "NG"
    :param image_path: 图片文件的完整路径
    :return: "OK" 或 "NG"
    """
    ok_threshold, collect_enabled = read_threshold_from_config()

    # 确保是图片文件（支持 .jpg 和 .png）
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        logging.info(f"Warning: {image_path} is not a valid image file. Skipping...")
        return "NG"

    # 先检查模板匹配是否为 OK
    if template_match(image_path, './temp', ok_threshold) == "OK":
        result = "OK"
        logging.info(f"图片{image_path}判定结果: {result}")
        return result

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        logging.info(f"Warning: Unable to read image {image_path}. Skipping...")
        return "NG"

    # 获取原始图片尺寸
    original_height, original_width, _ = image.shape

    # 调整图片大小，保持宽高比，最大边为 640
    max_size = 640
    scale = min(max_size / original_width, max_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建一个 640x640 的白色画布，并将调整后的图片居中放置
    padded_image = np.ones((max_size, max_size, 3), dtype=np.uint8) * 255  # 白色背景
    start_x = (max_size - new_width) // 2
    start_y = (max_size - new_height) // 2
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    # 计算图片中心点坐标
    center_x, center_y = max_size // 2, max_size // 2  # 图片中心点坐标

    # 定义中心范围的阈值（10%）
    threshold_x = int(max_size * 0.1)
    threshold_y = int(max_size * 0.1)

    # 使用模型进行推理
    results = model(padded_image)

    # 设置一个标志变量，用于判断是否满足条件
    ok_flag = False

    # 遍历检测结果
    for result in results:
        boxes = result.boxes  # 获取检测框信息
        for box in boxes:
            # 获取检测框的坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 获取类别索引和置信度
            class_id = int(box.cls[0])  # 类别索引
            confidence = float(box.conf[0])  # 置信度

            ok_range = 0.8
            if class_id == 5:
                confidence_threshold = ok_range
            elif class_id == 2:
                confidence_threshold = 0.5
            else:
                confidence_threshold = 0.6

            # 如果置信度低于阈值，跳过该检测框
            if confidence < confidence_threshold:
                continue

            # 计算检测框的中心点
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                ok_flag = True  # 满足条件
            # 判断检测框中心点是否在图片中心 10% 范围内
            if abs(box_center_x - center_x) <= threshold_x and abs(box_center_y - center_y) <= threshold_y:
                ok_flag = True  # 满足条件

    # 返回预测结果
    prediction_result = "OK" if ok_flag else "NG"
    logging.info(f"图片{image_path}判定结果: {prediction_result}")

    return prediction_result

def process_bzmd_file(input_path):
    history_folder = os.path.join(os.getcwd(), "history")
    ok_folder = os.path.join(history_folder, "OK")
    ng_folder = os.path.join(history_folder, "NG")
    ok_threshold, collect_enabled = read_threshold_from_config()
    
    # 创建 history 文件夹及子文件夹
    os.makedirs(ok_folder, exist_ok=True)
    os.makedirs(ng_folder, exist_ok=True)
    
    """直接在原文件上修改（通过临时文件替换）"""
    temp_path = input_path + '.tmp'
    all_ok = True  # 标志变量，表示所有的 predict 结果是否全为 OK
    image_dir = os.path.dirname(input_path)  # 获取 bzmd 文件所在的文件夹路径

    task_order = ""
    program_name = ""
    board_code = ""

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(temp_path, 'w', encoding='utf-8') as outfile:
        first_line = infile.readline()
        if first_line.lstrip().startswith('<inspectResult appData='):
            app_data_match = re.search(r'appData="([^"]+)"', first_line)
            if app_data_match:
                app_data_str = app_data_match.group(1)
                app_dict = {}
                for pair in app_data_str.split(';'):
                    if pair:
                        key, val = pair.split(',', 1)
                        app_dict[key] = val

                task_order = app_dict.get('LotName', "")
                program_name = app_dict.get('b_name', "")
                board_code = app_dict.get('Date', "")

        outfile.write(first_line)
        
        for line in infile:
            if line.lstrip().startswith('<inspectWindowResult appData=') and 'Ecd' in line:
                app_data_match = re.search(r'appData="([^"]+)"', line)
                if not app_data_match:
                    outfile.write(line)
                    continue

                app_data_str = app_data_match.group(1)
                app_dict = {}
                for pair in app_data_str.split(';'):
                    if pair:
                        key, val = pair.split(',', 1)
                        app_dict[key] = val

                bitmap = app_dict.get('Bitmap', '')
                if not bitmap:
                    outfile.write(line)
                    continue

                # 拼接完整图片路径
                image_path = os.path.join(image_dir, bitmap.replace(".JPG", "_Color.JPG"))
                prediction = predict(image_path)

                copy_to_display_folder(image_path, prediction)
                # 获取图片文件名
                image_name = os.path.basename(image_path)
                # 写入历史记录

                write_to_history(task_order, program_name, board_code, image_name, "Ecd", prediction)
                if prediction != 'OK':
                    all_ok = False  # 如果有非 OK 结果，则标记为 False
                    if collect_enabled == 1:
                        copy_image_with_suffix(image_path, ng_folder, prediction)

                if prediction == 'OK':  # 如果是误判
                    current_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    app_dict['actionFlg_0'] = '3'  # 修改 actionFlg_0
                    ai_ng_type = app_dict.get('AiNgType_0', 'MISSING')
                    new_fields = []
                    for pair in app_data_str.split(';'):
                        if not pair:
                            continue
                        key = pair.split(',', 1)[0]
                        if key == 'actionFlg_0':
                            new_fields.append(f'actionFlg_0,3')
                        elif key == 'AiNgType_0':
                            new_fields.append(pair)
                            new_fields.append(f'OpNgType_0,{ai_ng_type}')
                        elif key != 'operator_0':
                            new_fields.append(pair)

                    disp_order_idx = next((i for i, s in enumerate(new_fields)
                                          if s.startswith('dispOrder,')), len(new_fields))
                    new_fields.insert(disp_order_idx, f'reviseDate_0,{current_time}')
                    new_fields.insert(disp_order_idx + 1, 'operator_0,Administrator')

                    new_app_str = ';'.join(new_fields)
                    new_line = re.sub(r'(appData=")[^"]+(")', rf'\g<1>{new_app_str}\g<2>', line)
                    outfile.write(new_line)
                    if collect_enabled == 1:
                        copy_image_with_suffix(image_path, ok_folder, prediction)
                else:
                    outfile.write(line)
            elif line.lstrip().startswith('<inspectWindowResult appData='):
                # 拼接完整图片路径
                image_path = os.path.join(image_dir, bitmap.replace(".JPG", "_Color.JPG"))
                # 获取图片文件名
                image_name = os.path.basename(image_path)
                # 写入历史记录
                write_to_history(task_order, program_name, board_code, image_name, "非ecd", "非ECD未判定")
                all_ok = False
                logging.info("非ECD已跳过")
                
                outfile.write(line)
            else:
                outfile.write(line)
    
    # 替换原文件
    os.replace(temp_path, input_path)
    logging.info("bzmd文件修改完成 %s", all_ok)
    return all_ok

def process_bfre_file(input_path):
    """直接在原文件上修改（通过临时文件替换）"""
    temp_path = input_path + '.tmp'
    logging.info("开始处理bfre文件")
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(temp_path, 'w', encoding='utf-8') as outfile:
        section = None
        auxflag_added = False

        for line in infile:
            stripped_line = line.strip()
            if stripped_line in ('[AOI_DATA]', '[ACTION_INFO]', '[JUDGEMENT_DATA]'):
                section = stripped_line.strip('[]')
            
            if section == 'ACTION_INFO':
                if line.startswith('ERRORCODE='):
                    outfile.write('ERRORCODE=\n')
                    if not auxflag_added:
                        outfile.write('AUXFLAG=None\n')
                        auxflag_added = True
                elif line.startswith('ACTION_FLAG='):
                    outfile.write('ACTION_FLAG=FalseCall\n')  # 修改 ACTION_FLAG
                elif line.startswith('IS_JUDGED='):
                    outfile.write('IS_JUDGED=True\n')  # 添加 IS_JUDGED=True
                else:
                    outfile.write(line)
            elif section == 'JUDGEMENT_DATA':
                if line.startswith('JUDGEMENT_RESULT='):
                    outfile.write('JUDGEMENT_RESULT=FalseCall\n')  # 修改 JUDGEMENT_RESULT
                elif line.startswith('JUDGEMENT_TIME='):
                    current_time = datetime.now().strftime('%Y%%2F%m%%2F%d%%20%H%%3A%M%%3A%S')
                    outfile.write(f'JUDGEMENT_TIME={current_time}\n')
                elif line.startswith('OPERATOR_NAME='):
                    outfile.write('OPERATOR_NAME=Administrator\n')  # 修改 OPERATOR_NAME
                elif line.startswith('OPERATOR_GUID='):
                    outfile.write('OPERATOR_GUID=f17b100c-8908-4e92-aff9-2c6012065041\n')  # 添加固定的 OPERATOR_GUID
                elif line.startswith('COMMENT='):
                    continue
                else:
                    outfile.write(line)
            else:
                outfile.write(line)
    
    # 替换原文件
    os.replace(temp_path, input_path)
    logging.info("bfre文件修改完成")

def is_ok_board(bzmd_file):
    try:
        with open(bzmd_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return 'aiJudgement="Ok"' in first_line
    except Exception as e:
        logging.error(f"读取文件失败: {bzmd_file}, 错误: {str(e)}")
        return False


def process_all_files(check, input_root, output_target):
    logging.info("文件生成路径：%s", output_target)
    logging.info("文件扫描路径：%s", input_root)

    while check:
        for root, dirs, files in os.walk(input_root):
            bzmd_files = [f for f in files if f.endswith('.bzmd')]
            bfre_files = [f for f in files if f.endswith('.bfre')]

            if not bzmd_files or not bfre_files:
                continue

            bzmd_file = os.path.join(root, bzmd_files[0])
            bfre_file = os.path.join(root, bfre_files[0])
            logging.info(f"扫描到{bzmd_file}")

            # 判断是否是 OK 板
            if is_ok_board(bzmd_file):
                logging.info(f"{bzmd_file} 是 OK 板（包含 aiJudgement=\"Ok\"），跳过处理并直接移动")

                rel_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_target, rel_path)
                task_order = ""
                program_name = ""
                board_code = ""
                try:
                    with open(bzmd_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        app_data_match = re.search(r'appData="([^"]+)"', first_line)
                        if app_data_match:
                            app_data_str = app_data_match.group(1)
                            app_dict = {}
                            for pair in app_data_str.split(';'):
                                if pair:
                                    key, val = pair.split(',', 1)
                                    app_dict[key] = val
                            task_order = app_dict.get('LotName', "")
                            program_name = app_dict.get('b_name', "")
                            board_code = app_dict.get('Date', "")
                except Exception as e:
                    logging.error(f"读取 appData 失败: {str(e)}")

                # 写入历史记录
                write_to_history(task_order, program_name, board_code, "NONE", "NONE", "NONE")

                rel_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_target, rel_path)

                # 删除旧目录（如果存在）
                if os.path.exists(target_dir):
                    try:
                        shutil.rmtree(target_dir)
                        logging.info(f"已删除旧的目标文件夹：{target_dir}")
                    except Exception as e:
                        logging.error(f"删除旧文件夹失败：{target_dir}，错误：{str(e)}")
                        continue

                # 移动文件夹
                try:
                    shutil.move(root, target_dir)
                    logging.info(f"Processed and moved {root} → {target_dir}")
                except Exception as e:
                    logging.error(f"移动文件夹失败：{root} → {target_dir}，错误：{str(e)}")
                continue  # 跳过后续处理

            # 非 OK 板，按原逻辑处理
            start_time = time.time()
            all_ok = process_bzmd_file(bzmd_file)

            if all_ok:
                process_bfre_file(bfre_file)

            rel_path = os.path.relpath(root, input_root)
            target_dir = os.path.join(output_target, rel_path)

            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                    logging.info(f"已删除旧的目标文件夹：{target_dir}")
                except Exception as e:
                    logging.error(f"删除旧文件夹失败：{target_dir}，错误：{str(e)}")
                    continue

            try:
                shutil.move(root, target_dir)
                logging.info(f"Processed and moved {root} → {target_dir}")
            except Exception as e:
                logging.error(f"移动文件夹失败：{root} → {target_dir}，错误：{str(e)}")

            if all_ok:
                elapsed = time.time() - start_time
                delay = max(0, 10 - elapsed)
                if delay > 0:
                    logging.info(f"等待 {delay:.2f} 秒后过板...")
                    time.sleep(delay)

                icon_file = "img/click.jpg"
                if not os.path.exists(icon_file):
                    logging.info("请设置过板图标img/click.jpg，并手动过板")
                    continue
                success = find_and_click_icon(icon_file, confidence=0.7)
                if success:
                    logging.info("已自动过板")
                else:
                    logging.error("未找到过板图标，请手动过板")

            logging.info("复判完成")

        time.sleep(3)


