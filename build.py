import subprocess
import os
import shutil


#set PYTHONPATH=C:\Users\z00588771\Anaconda3\envs\ocr;C:\Users\z00588771\Anaconda3\envs\ocr\Lib\site-packages;D:\AIjudege\new
# 配置信息
APP_NAME = "AOI智能复判系统2.0"
MAIN_SCRIPT = "main.py"
ICON_FILE = os.path.join("img", "icon.ico")  
MODULES_TO_HIDE_IMPORT = [
    "shenzhou",
    "benchuangSMT",
    "benchuang",
    "saki",
    "shenzhouSMT",
    "ky",
    "matplotlib.backends.backend_tkagg",
]

# 检查图标文件是否存在
if not os.path.exists(ICON_FILE):
    print(f"❌ 错误：图标文件 {ICON_FILE} 不存在！请准备一个 icon.ico 放在 img 文件夹中。")
    exit(1)

# 构建 PyInstaller 参数
cmd = [
    'pyinstaller',
    '--clean',
    '--noconfirm',
    "--noconsole",
    '--name=AOI智能复判系统2.0',
    '--icon=img\\icon.ico',
    '--collect-all', 'paddleocr',
    '--collect-all', 'pypdfium2',
    '--collect-all', 'paddlex',
    '--collect-all', 'cv2',
    '--collect-all', 'paddle',
]

# 添加 hidden-import
for module in MODULES_TO_HIDE_IMPORT:
    cmd.append(f"--hidden-import={module}")

# 添加主程序
cmd.append(MAIN_SCRIPT)

# 执行打包命令
print("🚀 正在执行打包命令：")
print(" ".join(cmd))

try:
    subprocess.check_call(cmd)
    
    # 定义输出路径
    output_dir = os.path.join("dist", APP_NAME)
    
    # 要复制的资源文件夹
    folders_to_copy = ["img", "model", "temp"]

    # 复制每个文件夹到输出目录
    for folder in folders_to_copy:
        src_path = folder
        dest_path = os.path.join(output_dir, folder)
        if os.path.exists(src_path):
            if os.path.exists(dest_path):
                print(f"🔄 删除已存在的目标文件夹: {dest_path}")
                shutil.rmtree(dest_path)
            print(f"📁 正在复制 {src_path} 到 {dest_path}")
            shutil.copytree(src_path, dest_path)
        else:
            print(f"⚠️ 未找到源文件夹: {src_path}，跳过复制")

    print("\n✅ 打包并资源复制已完成！位于 dist/ 目录下")
except subprocess.CalledProcessError as e:
    print(f"\n❌ 打包失败: {e}")