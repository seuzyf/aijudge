import torch
import os
import sys
from PIL import Image
from torchvision import transforms
import time
import logging
import shutil
import torch.nn as nn
from torchvision import models
from datetime import datetime
import csv

logging.basicConfig(filename='check.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
name2label = {'NG': 0, 'OK': 1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize = 224

# 写入历史记录函数
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

tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),  # string path => image data
    transforms.Resize((int(resize * 1.25), int(resize * 1.25))),  # 数据预处理部分
    transforms.RandomRotation(15),
    transforms.CenterCrop(resize),  # 防止旋转后边界出现黑框部分
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = VGG16net().to(device)
model.load_state_dict(torch.load('./model/shenzhou.pth'))
model.eval()

def read_threshold_from_config(config_path='config.txt'):
    """
    从 config.txt 文件中读取 OKrange 和 collect 配置。
    如果文件不存在或未指定字段，则创建文件并写入默认值。
    """
    default_okrange = 0.9
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

def copy_image_with_suffix(src_path, dest_folder, result, probability):
    """
    复制图片到目标文件夹，避免重名问题。
    将分类结果和概率值添加到文件名中，格式为：原文件名_结果_概率后缀.jpg
    如果目标文件夹中已存在同名图片，则自动添加编号后缀。
    :param src_path: 源图片路径
    :param dest_folder: 目标文件夹路径
    :param result: 分类结果（OK/NG）
    :param probability: 分类概率值（0-1之间的浮点数）
    """
    # 格式化概率值（保留4位小数，替换小数点为下划线）
    prob_str = f"{probability:.4f}".replace('.', '_')
    # 分离文件名和扩展名
    base_name, ext = os.path.splitext(os.path.basename(src_path))
    # 构建基础文件名（包含结果和概率）
    base_filename = f"{base_name}_{result}_{prob_str}{ext}"
    dest_path = os.path.join(dest_folder, base_filename)
    # 处理文件名冲突
    counter = 1
    while os.path.exists(dest_path):
        # 在概率后添加数字后缀
        new_filename = f"{base_name}_{result}_{prob_str}_{counter}{ext}"
        dest_path = os.path.join(dest_folder, new_filename)
        counter += 1
    # 执行复制操作
    shutil.copy(src_path, dest_path)
    logging.info(f"图片已复制到 {dest_path} "
                 f"（分类：{result}，概率：{probability:.4f}）")


def process_all_files(check, directory, resPath):
    # 支持的图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png']
    imagePath = os.path.join(directory, '1')
    txtPath = os.path.join(directory, '3')
    distPath = os.path.join(directory, '4')
    collectPath = os.path.join(directory, '6')  # 图片收集路径
    # 创建收集文件夹
    os.makedirs(os.path.join(collectPath, 'OK'), exist_ok=True)
    os.makedirs(os.path.join(collectPath, 'NG'), exist_ok=True)
    logging.info("标识文件扫描路径：%s,", txtPath)
    logging.info("标识文件生成路径：%s,", distPath)
    logging.info("结果文件扫描路径：%s,", resPath)

    # 读取阈值和收集配置
    ok_threshold, collect_enabled = read_threshold_from_config()
    ng_threshold = 1 - ok_threshold  # NG 的阈值为 1 - OKrange

    while check:
        for filename in os.listdir(txtPath):
            if filename.endswith(".txt"):
                txt_path = os.path.join(txtPath, filename)
                with open(txt_path, 'r') as file:
                    lines = file.readlines()
                    first_line = lines[0]
                    second_line = lines[1]
                    imagedir = os.path.join(imagePath, second_line)
                    image_dir = imagedir.replace("\n", "").replace("\r", "")
                    output_txt_path = os.path.join(resPath, filename)

                    # 图片复判
                    if os.path.exists(image_dir):
                        # 遍历目录中的所有文件
                        logging.info("开始复判")
                        with open(output_txt_path, 'a') as file:
                            file.write(first_line)

                        for file_name in os.listdir(image_dir):
                            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                                img_path = os.path.join(image_dir, file_name)
                                imgname = os.path.splitext(os.path.basename(img_path))[0]
                                logging.info("开始复判图片%s,", imgname)

                                with torch.no_grad():
                                    img = tf(img_path).unsqueeze(0)
                                    img_ = img.to(device)
                                    outputs = model(img_)
                                    probabilities = torch.softmax(outputs, dim=1)
                                    # 获取OK和NG的概率
                                    ng_probability = probabilities[:, 0].item()
                                    ok_probability = probabilities[:, 1].item()
                                    prob = 0
                                    # 根据阈值判断结果
                                    if ok_probability >= ok_threshold:
                                        result = "OK"
                                        prob = ok_probability
                                    else:
                                        result = "NG"
                                        prob = ng_probability

                                # 写入复判结果
                                copy_to_display_folder(img_path, result)
                                with open(output_txt_path, 'a') as file:
                                    file.write(f"{file_name},{result}\n")
                                logging.info(f"图片 {file_name}，复判结果为: {result}")

                                # 如果开启收集功能，则复制图片
                                if collect_enabled == 1:
                                    result_folder = os.path.join(collectPath, result)
                                    copy_image_with_suffix(img_path, result_folder, result, prob)

                                # 解析第二行数据获取程序名、单板条码等信息
                                parts = second_line.strip().rstrip('\\').split('\\')

                                if len(parts) >= 4:
                                    program_name = parts[1]  # 程序名

                                    last_part = parts[-1]  # AOI710_1024A6350268_101436004
                                    segments = last_part.split('_')

                                    if len(segments) >= 3:
                                        board_code = segments[1]  # 单板条码中间部分
                                    else:
                                        board_code = "Unknown"
                                else:
                                    program_name = "Unknown"
                                    board_code = "Unknown"

                                # 解析图片名中的缺陷类型
                                image_parts = file_name.split(",")
                                if len(image_parts) >= 3:
                                    ngtype = image_parts[-1].split(".")[0]  # 缺陷类型
                                else:
                                    ngtype = "Unknown"
                                # 调用写入历史函数
                                try:
                                    write_to_history(
                                        task_order="Unknown",  # 任务令暂时全写\
                                        program_name=program_name,
                                        board_code=board_code,
                                        image_name=file_name,
                                        ngtype=ngtype,
                                        result=result
                                    )
                                except Exception as e:
                                    logging.error(f"写入历史记录异常: {e}", exc_info=True)


                        end_path = os.path.join(distPath, filename)
                        with open(end_path, 'w') as file:
                            file.write(first_line)

                logging.info("删除%s,", txt_path)
                os.remove(txt_path)
                logging.info("复判完成，5S后启动下次扫描")

        # 每隔2秒检查一次
        time.sleep(2)


