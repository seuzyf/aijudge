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
import pickle
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(filename='check.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
name2label = {'NG': 0, 'OK': 1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize = 256  
DISPLAY_FOLDER = "display"
os.makedirs(DISPLAY_FOLDER, exist_ok=True)
disp = 1

# ================= CPU 性能全局优化设置 =================
optimal_threads = max(1, os.cpu_count() // 2)
torch.set_num_threads(optimal_threads)
logging.info(f"PyTorch CPU 计算线程数已优化为: {optimal_threads}")

# ================= 核心模型替换：PatchCore 极速特征提取器 =================
class FeatureExtractor(nn.Module):
    def __init__(self, weight_path):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.wide_resnet50_2(weights=None)
        
        try:
            self.backbone.load_state_dict(torch.load(weight_path, map_location='cpu'))
            logging.info(f"成功加载特征提取骨干网络: {weight_path}")
        except Exception as e:
            logging.error(f"加载骨干网络失败，请检查路径: {e}")
            
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.stage0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.stage1 = self.backbone.layer1 
        self.stage2 = self.backbone.layer2 

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            x = self.stage0(x)
            f1 = self.stage1(x)    
            f2 = self.stage2(f1) 
            
            f2 = nn.functional.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
            feat = torch.cat([f1, f2], dim=1) 
            
            # 【实战调优 1：3x3 物理防抖池化】抹平流水线机械位移误差，断绝良品分数倒挂
            feat = nn.functional.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            
            # 【实战调优 2：通道无损切片】768 维等距抽帧压缩至 384 维，推演算力飙升一倍
            feat = feat[:, ::2, :, :]
            
            feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat

# ================= 预处理修改 =================
tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'), 
    transforms.Resize((resize, resize)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= 加载模型与记忆词典 =================
MODEL_WEIGHT_PATH = './model/yw.pth'
MEMORY_BANK_PATH = './model/yw.pkl'

model = FeatureExtractor(MODEL_WEIGHT_PATH).to(device)
model = model.to(memory_format=torch.channels_last)
model.eval()

knn = None
gallery_tensor = None
n_neighbors = 5 

try:
    with open(MEMORY_BANK_PATH, 'rb') as f:
        knn = pickle.load(f)
    logging.info(f"成功加载 PatchCore 记忆词典: {MEMORY_BANK_PATH}")
    
    # 强制将内存转换为连续格式，避免矩阵乘法跨内存页导致 CPU 变慢
    if hasattr(knn, '_fit_X'):
        gallery_tensor = torch.from_numpy(knn._fit_X).float().to(device).T.contiguous()
        n_neighbors = knn.n_neighbors
        logging.info(f"已构建高速矩阵检索域，特征维度: {gallery_tensor.shape}")
        
except Exception as e:
    logging.error(f"加载记忆词典失败，系统无法进行推演: {e}")

# ================= 原有的辅助函数 =================
def copy_to_display_folder(image_path, result):
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
    default_lag = 1.0 
    
    okrange_found, collect_found, lag_found = False, False, False
    okrange_value, collect_value, lag_value = default_okrange, default_collect, default_lag

    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f"okRange={default_okrange}\n")
                f.write(f"collect={default_collect}\n")
                f.write(f"lag={default_lag}\n")
            return default_okrange, default_collect, default_lag
            
        with open(config_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or '=' not in line:
                    continue
                if line.startswith("okRange"):
                    _, val = line.split("=")
                    okrange_value = float(val)
                    okrange_value = max(0.0, min(1.0, okrange_value)) 
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
            if not okrange_found: f.write(f"okRange={default_okrange}\n")
            if not collect_found: f.write(f"collect={default_collect}\n")
            if not lag_found: f.write(f"lag={default_lag}\n")
                
        return okrange_value, collect_value, lag_value
    except Exception as e:
        logging.error(f"配置读取失败: {e}")
        return default_okrange, default_collect, default_lag

def copy_image_with_suffix(src_path, dest_folder, ngtype, result):
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
    logging.info(f"图片已归档复制到 {dest_path}")

def write_to_history(task_order, program_name, board_code, image_name, ngtype, result):
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
        
    with open(csv_file, mode='r', newline='', encoding='utf-8-sig') as f:
        first_line = f.readline().strip()
        
    if first_line != ','.join(expected_header):
        backup_file = csv_file + '.bak'
        os.rename(csv_file, backup_file)
        with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as fout:
            writer = csv.writer(fout)
            writer.writerow(expected_header)
            with open(backup_file, mode='r', newline='', encoding='utf-8-sig') as fin:
                for line in fin:
                    fout.write(line)
        logging.warning(f"【警告】检测到文件无有效表头，已自动补全，并从 {backup_file} 恢复数据")
        
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

# ================= 主流程控制 =================
def process_all_files(check, directory, xmlPath):
    logger = logging.getLogger()
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('check.log', mode='a', encoding='gbk')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"智能电动异物检测模块已启动，监控主路径: {directory}")
    logging.info(f"当前复判xml读取路径: {xmlPath}")
    
    parent_dir = os.path.dirname(directory)
    historyPath = os.path.join(parent_dir, 'history')
    okPath = os.path.join(historyPath, 'OK')
    ngPath = os.path.join(historyPath, 'NG')
    resPath = os.path.join(parent_dir, 'AI')
    
    os.makedirs(historyPath, exist_ok=True)
    os.makedirs(okPath, exist_ok=True)
    os.makedirs(ngPath, exist_ok=True)
    os.makedirs(resPath, exist_ok=True)
    
    global disp
    disp = 1
    os.makedirs(DISPLAY_FOLDER, exist_ok=True)
    processed_folders = set()

    while check:
        if not os.path.exists(directory):
            time.sleep(2)
            continue

        items = os.listdir(directory)
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        pending_folders = [f for f in folders if f not in processed_folders and f not in ['history', 'AI', 'display']]

        if not pending_folders:
            time.sleep(2)
            continue

        current_folder_name = pending_folders[0]
        okrange_strictness, collect, lag = read_threshold_from_config()
        
        # 【判断阈值】
        base_threshold = 0.4410 
        threshold_margin = 0.4
        patchcore_threshold = base_threshold + (0.5 - okrange_strictness) * threshold_margin
        
        logging.info(f"扫描到复判文件夹: {current_folder_name}，等待 {lag} 秒输出...")
        time.sleep(lag)
        
        checkPath = os.path.join(directory, current_folder_name) 
        checkPath_nested = os.path.join(checkPath, current_folder_name) 
        if os.path.exists(checkPath_nested) and os.path.isdir(checkPath_nested):
            checkPath = checkPath_nested

        imagePath = os.path.join(checkPath, 'NGPartImage')
        csv_files = [f for f in os.listdir(checkPath) if f.endswith(".csv")]
        
        if not csv_files:
            processed_folders.add(current_folder_name)
            continue

        for filename in csv_files:
            csvPath = os.path.join(checkPath, filename)
            try:
                with open(csvPath, 'rb') as f:
                    content_bytes = f.read()
                try:
                    decoded_content = content_bytes.decode('utf-8-sig')
                except UnicodeDecodeError:
                    decoded_content = content_bytes.decode('gbk', errors='ignore')
                df = pd.read_csv(io.StringIO(decoded_content))
            except Exception as e:
                logging.error(f"读取CSV文件 {csvPath} 失败: {e}")
                continue 
                
            logging.info(f"CSV读取成功，共 {len(df)} 行数据准备进行复判。")

            for index, row in df.iterrows():
                program_name = row.get('JOBNAME', '')   
                board_code = row.get('ARRAY_BARCODE', '')  
                ngtype = row.get('NG_NAME', '')         
                
                try:
                    file_name_part = f"{row.iloc[4]}@{row.iloc[7]}"  
                except IndexError:
                    file_name_part = f"{row.iloc[4]}" if len(row) > 4 else "unknown"

                suffix_list = [".jpg", ".JPG", "_AC.jpg", "_ac.jpg", "_AC.JPG", "_AC.png", ".png"]
                
                image_path = None
                for suffix in suffix_list:
                    temp_path = os.path.join(imagePath, f"{file_name_part}{suffix}")
                    if os.path.exists(temp_path):
                        image_path = temp_path
                        break
                        
                if not image_path:
                    df.at[index, 'FLAG'] = 'AING'
                    continue
                
                imgname = os.path.splitext(os.path.basename(image_path))[0]
                col_g_val = str(row.iloc[6]) if len(row) > 6 else ""
                
                is_target = col_g_val.lower().startswith('partcode')
                if is_target:
                    ngtype = col_g_val  
                    logging.info(f"开始推断图片 {imgname}, 类别:{ngtype}")
                    
                    if knn is None:
                        logging.error("严重错误：未加载特征词典，默认输出 AING")
                        checkresult, res = "AING", "NG"
                    else:
                        with torch.inference_mode(): 
                            img = tf(image_path).unsqueeze(0).to(device).to(memory_format=torch.channels_last)
                            feats = model(img)
                            
                            B, C, H, W = feats.shape
                            feats = feats.contiguous().view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
                            
                            if gallery_tensor is not None:
                                batch_size = 1024
                                max_distances = []
                                
                                for i in range(0, feats.shape[0], batch_size):
                                    chunk = feats[i:i+batch_size]
                                    sim = torch.matmul(chunk, gallery_tensor)
                                    dist_sq = 2.0 - 2.0 * sim
                                    dist_sq = torch.clamp(dist_sq, min=0.0)
                                    dist = torch.sqrt(dist_sq)
                                    
                                    topk_dist, _ = torch.topk(dist, n_neighbors, dim=1, largest=False)
                                    mean_dist = topk_dist.mean(dim=1)
                                    max_distances.append(mean_dist.max().item())
                                    
                                anomaly_score = float(max(max_distances))
                            else:
                                distances, _ = knn.kneighbors(feats.cpu().numpy())
                                anomaly_score = float(distances.mean(axis=1).max())
                            
                            if anomaly_score < patchcore_threshold:
                                checkresult, res = "AIOK", "OK"
                                logging.info(f"【判定 OK】 异常分: {anomaly_score:.4f} < 阈值: {patchcore_threshold:.4f} (严格度:{okrange_strictness})")
                            else:
                                checkresult, res = "AING", "NG"
                                logging.info(f"【判定 NG】 异常分: {anomaly_score:.4f} >= 阈值: {patchcore_threshold:.4f} (严格度:{okrange_strictness})")

                    write_to_history('Unknown', program_name, board_code, imgname, ngtype, res)
                    copy_to_display_folder(image_path, checkresult)
                    
                    if collect == 1:
                        dest_folder = ngPath if checkresult == "AING" else okPath
                        refid = row.get('REFID', '')        
                        copy_image_with_suffix(image_path, dest_folder, ngtype, refid)
                        
                    df.at[index, 'FLAG'] = checkresult
                else:
                    df.at[index, 'FLAG'] = "AING"

            try:
                df.to_csv(csvPath, index=False)
                shutil.copy(csvPath, resPath)
                shutil.copy(csvPath, historyPath)
                os.remove(csvPath)
                logging.info(f"CSV文件已保存并归档: {csvPath}")
            except Exception as e:
                logging.error(f"处理 CSV 归档时出错: {e}")

        processed_folders.add(current_folder_name)
        logging.info(f"文件夹 {current_folder_name} 的复判流程全部结束。")
    time.sleep(2)