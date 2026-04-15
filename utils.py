import importlib

DEVICE_MODULE_MAP = {
    "神州": "shenzhou",
    "神州SMT": "shenzhouSMT",
    "奔创SMT": "benchuangSMT",
    "奔创": "benchuang",
    "Saki": "saki",
    "KY": "ky"
}

def get_device_module(device_name):
    """
    根据设备名称返回对应的模块。
    :param device_name: 设备名，例如 "神州", "奔创", "Saki", "ky"
    :return: 导入的模块对象
    """
    module_name = DEVICE_MODULE_MAP.get(device_name)
    if not module_name:
        raise ImportError(f"不支持的设备类型: {device_name}")

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"无法导入模块 {module_name}: {e}")