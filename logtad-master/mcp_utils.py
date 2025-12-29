"""
MCP 工具辅助函数模块
提供日志记录、错误处理、文件管理等辅助功能
"""
import os
import json
import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

# 配置日志
def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_dir: 日志目录路径
        
    Returns:
        Logger对象
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("LogTAD_MCP")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 文件处理器
    log_file = os.path.join(log_dir, "logtad_mcp.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

def load_config(config_path: str = "mcp_config.json") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"配置文件加载成功: {config_path}")
                return config
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return get_default_config()
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "host": "127.0.0.1",
        "port": 4003,
        "transport": "sse",
        "model_dir": "./saved_model",
        "data_dir": "./Dataset",
        "log_dir": "./logs"
    }

def ensure_dirs(dirs: list):
    """
    确保目录存在，不存在则创建
    
    Args:
        dirs: 目录路径列表
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"确保目录存在: {dir_path}")

def validate_dataset_name(dataset_name: str) -> bool:
    """
    验证数据集名称是否有效
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        是否有效
    """
    valid_datasets = ["BGL", "Thunderbird"]
    if dataset_name not in valid_datasets:
        logger.error(f"无效的数据集名称: {dataset_name}，有效值: {valid_datasets}")
        return False
    return True

def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 是否必须存在
        
    Returns:
        是否有效
    """
    if not file_path:
        logger.error("文件路径为空")
        return False
    
    if must_exist and not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    return True

def format_error_response(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    格式化错误响应
    
    Args:
        error: 异常对象
        context: 上下文信息
        
    Returns:
        错误响应字典
    """
    error_msg = str(error)
    error_type = type(error).__name__
    traceback_str = traceback.format_exc()
    
    logger.error(f"{context} - {error_type}: {error_msg}")
    logger.debug(f"详细错误信息:\n{traceback_str}")
    
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": error_msg,
            "context": context
        }
    }

def format_success_response(data: Any, message: str = "操作成功") -> Dict[str, Any]:
    """
    格式化成功响应
    
    Args:
        data: 响应数据
        message: 成功消息
        
    Returns:
        成功响应字典
    """
    return {
        "success": True,
        "message": message,
        "data": data
    }

def get_model_files(source_dataset: str, target_dataset: str, model_dir: str) -> Dict[str, str]:
    """
    获取模型相关文件路径
    
    Args:
        source_dataset: 源数据集名称
        target_dataset: 目标数据集名称
        model_dir: 模型目录
        
    Returns:
        模型文件路径字典
    """
    model_prefix = f"{source_dataset}-{target_dataset}"
    return {
        "model": os.path.join(model_dir, f"{model_prefix}.pt"),
        "center": os.path.join(model_dir, f"{model_prefix}_center.csv"),
        "w2v": os.path.join(model_dir, f"{model_prefix}_w2v.bin")
    }

def check_model_exists(source_dataset: str, target_dataset: str, model_dir: str) -> bool:
    """
    检查模型文件是否存在
    
    Args:
        source_dataset: 源数据集名称
        target_dataset: 目标数据集名称
        model_dir: 模型目录
        
    Returns:
        模型是否存在
    """
    model_files = get_model_files(source_dataset, target_dataset, model_dir)
    for file_path in model_files.values():
        if not os.path.exists(file_path):
            logger.warning(f"模型文件不存在: {file_path}")
            return False
    return True

def get_device() -> str:
    """
    获取可用的设备 (CUDA/CPU)
    
    Returns:
        设备名称
    """
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("使用CPU")
    return device

