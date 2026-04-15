import cv2
import math
import os
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import QApplication, QWidget
import time
import pandas as pd
import shutil
import logging
import numpy as np
from datetime import datetime
import csv
import io
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models

logging.basicConfig(filename='check.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
name2label = {'NG': 0, 'OK': 1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize = 224
DISPLAY_FOLDER = "display"
os.makedirs(DISPLAY_FOLDER, exist_ok=True)
disp = 1

# 冻结网络层的参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False 


# 定义 CBAM 注意力模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# VGG 网络结构定义
class VGG16net(nn.Module):
    def __init__(self, feature_extract=True, num_class=2):
        checkpoint_path = "./model/vgg16-397923af.pth"
        super(VGG16net, self).__init__()
        model = models.vgg16()
        model.load_state_dict(torch.load(checkpoint_path))
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)
        # 解冻最后几层卷积层
        for param in self.features[-3:].parameters():
            param.requires_grad = True
        self.cbam = CBAM(512)  # 添加 CBAM 注意力模块
        self.avgpool = model.avgpool
        # 更深的分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),  # 批量归一化
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)  # 应用注意力模块
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        out = self.classifier(x)
        return out

tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),  # string path => image data
    transforms.Resize((int(resize * 1.25), int(resize * 1.25))),  # 数据预处理部分
    transforms.RandomRotation(15),
    transforms.CenterCrop(resize),  # 防止旋转后边界出现黑框部分
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = VGG16net().to(device)
model.load_state_dict(torch.load('./model/benchuang.pth'))
model.eval()

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
    okrange, collect = read_threshold_from_config()

    while check:
        items = os.listdir(directory)
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        if not folders:
            time.sleep(2)
            continue

        #  获取当前处理的文件夹名
        current_folder_name = folders[0]
        checkPath = os.path.join(directory, current_folder_name) # 使用 current_folder_name
        checkPath = os.path.join(checkPath, current_folder_name) # 使用 current_folder_name
        imagePath = os.path.join(checkPath, 'NGPartImage')

        for filename in os.listdir(checkPath):
            if not filename.endswith(".csv"):
                continue

            csvPath = os.path.join(checkPath, filename)
            file_name, _ = os.path.splitext(filename)
            # 不再创建子文件夹，直接使用OK和NG文件夹
            os.makedirs(okPath, exist_ok=True)
            os.makedirs(ngPath, exist_ok=True)
            logging.info("333。")
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

            for index, row in df.iterrows():
                # 从CSV中提取必要信息
                program_name = row.get('JOBNAME', '')   # 程序名
                board_code = row.get('ARRAY_BARCODE', '')  # 单板条码
                ngtype = row.get('NG_NAME', '').lower()         # 缺陷类型
                file_name_part = f"{row.iloc[4]}@{row.iloc[7]}"  # 使用iloc避免FutureWarning
                # 定义可能的后缀列表（按优先级排序）
                suffix_list = [
                    ".jpg",       # 其次尝试不带后缀的
                    ".JPG",       # 大写扩展名不带后缀
                    "_AC.jpg",    # 优先尝试带_AC后缀的
                    "_ac.jpg",    # 小写_ac后缀
                    "_AC.JPG",    # 大写扩展名
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
                    default_path = os.path.join(imagePath, f"{file_name_part}.jpg")  # 默认路径用于记录
                    logging.error(f"图片未找到: {default_path} (尝试了所有后缀: {suffix_list})")
                    print(f"图片未找到: {default_path} (尝试了所有后缀: {suffix_list})")
                    df.at[index, 'FLAG'] = 'AING'
                    continue
                
                imgname = os.path.splitext(os.path.basename(image_path))[0]

                logging.info(f"图片 {imgname},ngtype{ngtype}")
                if ngtype in ('s_no_solder', 'l_lift'):
                    checkresult = "AIOK"
                    res = "OK"
                    with torch.no_grad():
                        img = tf(image_path).unsqueeze(0)
                        img_ = img.to(device)
                        outputs = model(img_)
                        probabilities = torch.softmax(outputs, dim=1)
                        # 获取OK和NG的概率
                        ng_probability = probabilities[:, 0].item()
                        ok_probability = probabilities[:, 1].item()
                        prob = 0
                        # 根据阈值判断结果
                        if ok_probability >= okrange:
                            checkresult = "AIOK"
                            res = "OK"
                            logging.info(f"图片 {imgname},ok概率为 {ok_probability}，复判结果: OK")
                        else:
                            checkresult = "AING"
                            res = "NG"
                            logging.info(f"图片 {imgname},ok概率为 {ok_probability}，复判结果: NG")

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