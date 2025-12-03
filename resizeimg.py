import os
import cv2
import numpy as np
 
# 定义letterbox函数，用于对图像进行缩放和填充，使其适应指定尺寸
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=False, scale_fill=False, scale_up=False, stride=32):
    shape = img.shape[:2]  # 获取图像的原始高度和宽度
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # 如果new_shape是整数，则将其转换为方形尺寸
 
    # 计算缩放比例r，确保图像不会拉伸变形
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)  # 如果scale_up为False，则缩放比例不大于1，避免放大
 
    ratio = r, r  # 记录宽高的缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 根据比例计算缩放后的尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算需要填充的宽度和高度
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 如果auto为True，将填充尺寸调整为步长的倍数
    elif scale_fill:
        dw, dh = 0.0, 0.0  # 如果scale_fill为True，则不进行填充，直接拉伸到目标尺寸
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 重新计算缩放比例
 
    dw /= 2  # 左右填充的宽度
    dh /= 2  # 上下填充的高度
 
    if shape[::-1] != new_unpad:  # 如果原始尺寸和缩放后的尺寸不同，进行图像缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 线性插值进行图像缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 上下填充像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 左右填充像素数
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 使用指定颜色填充
    return img, ratio, (dw, dh)  # 返回处理后的图像、缩放比例以及填充的宽高
 
# 保存经过letterbox处理的图像
def save_letterboxed_image(image_bytes, output_filename, new_shape=(640, 640)):
    im0 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # 从字节数据解码出图像
    if im0 is None:
        raise ValueError("Failed to decode image from image_bytes")  # 如果图像解码失败，抛出异常
 
    img, _, _ = letterbox(im0, new_shape=new_shape)  # 对图像进行letterbox处理
 
    # 保存处理后的图像
    cv2.imwrite(output_filename, img)  # 将处理后的图像写入文件
    print(f"Saved letterboxed image to {output_filename}")  # 输出保存成功的信息
 
# 处理文件夹中的所有图像
def process_images_in_folder(folder_path, output_folder, new_shape=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果输出文件夹不存在，则创建
 
    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 仅处理特定格式的图像文件
            image_path = os.path.join(folder_path, filename)  # 获取图像的完整路径
            with open(image_path, 'rb') as f:
                image_bytes = f.read()  # 读取图像文件的字节数据
 
            output_filename = os.path.join(output_folder, filename)  # 构建输出文件的完整路径
            save_letterboxed_image(image_bytes, output_filename, new_shape=new_shape)  # 保存处理后的图像
 
# 使用示例
if __name__ == "__main__":
    folder_path = './'  # 替换为你的图片文件夹路径
    output_folder = './test'  # 替换为保存处理后图片的文件夹路径
    process_images_in_folder(folder_path, output_folder, new_shape=(640, 640))  # 处理文件夹中的图像