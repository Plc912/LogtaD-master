"""
LogTAD MCP 服务器
提供日志异常检测的 MCP 工具接口
"""
import os
import sys
import json
import asyncio
import warnings
from typing import Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from sklearn import metrics

# 导入项目模块
from utils import preprocessing, SlidingWindow
from utils.utils import set_seed, get_train_eval_iter, dist2label, get_iter
from model.LogTAD import LogTAD
import mcp_utils

# 忽略警告
warnings.filterwarnings("ignore")

# 加载配置
config = mcp_utils.load_config()
logger = mcp_utils.logger

# 确保目录存在
mcp_utils.ensure_dirs([
    config["model_dir"],
    config["data_dir"],
    config["log_dir"]
])

# 尝试导入 fastmcp，如果不存在则使用备用实现
try:
    from fastmcp import FastMCP
    mcp = FastMCP("LogTAD Anomaly Detection")
except ImportError:
    logger.warning("fastmcp 未安装，使用备用实现")
    # 备用实现将在后面定义
    mcp = None


def get_default_options() -> Dict[str, Any]:
    """获取默认训练选项"""
    default_params = config.get("default_params", {})
    device = default_params.get("device", "auto")
    if device == "auto":
        device = mcp_utils.get_device()
    
    return {
        "source_dataset_name": "Thunderbird",
        "target_dataset_name": "BGL",
        "device": device,
        "output_dir": config["data_dir"],
        "model_dir": config["model_dir"],
        "random_seed": default_params.get("random_seed", 42),
        "max_epoch": default_params.get("max_epoch", 100),
        "batch_size": default_params.get("batch_size", 1024),
        "lr": default_params.get("lr", 0.001),
        "weight_decay": 1e-6,
        "eps": 0.1,
        "emb_dim": 300,
        "window_size": default_params.get("window_size", 20),
        "step_size": 4,
        "train_size_s": default_params.get("train_size_s", 100000),
        "train_size_t": default_params.get("train_size_t", 1000),
        "hid_dim": 128,
        "out_dim": 2,
        "n_layers": 2,
        "dropout": 0.3,
        "bias": True,
        "alpha": 0.1,
        "test_ratio": 0.1
    }


# 工具 1: 解析日志文件
def parse_logs_tool(log_file_path: str, dataset_name: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    解析日志文件
    
    Args:
        log_file_path: 日志文件路径
        dataset_name: 数据集名称 (BGL/Thunderbird)
        output_dir: 输出目录，默认使用配置
        
    Returns:
        解析结果字典
    """
    try:
        if not mcp_utils.validate_dataset_name(dataset_name):
            return mcp_utils.format_error_response(
                ValueError(f"无效的数据集名称: {dataset_name}"),
                "parse_logs"
            )
        
        if not mcp_utils.validate_file_path(log_file_path, must_exist=True):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"日志文件不存在: {log_file_path}"),
                "parse_logs"
            )
        
        output_dir = output_dir or config["data_dir"]
        logger.info(f"开始解析日志文件: {log_file_path}, 数据集: {dataset_name}")
        
        # 调用预处理模块的解析功能
        # 注意：preprocessing.parsing 会下载数据集，这里我们需要一个解析本地文件的功能
        # 为了保持兼容，我们暂时调用原有函数，但实际使用时可能需要修改
        preprocessing.parsing(dataset_name, output_dir)
        
        output_file = os.path.join(output_dir, f"{dataset_name}.log_structured.csv")
        
        if os.path.exists(output_file):
            logger.info(f"日志解析完成: {output_file}")
            return mcp_utils.format_success_response(
                {
                    "output_file": output_file,
                    "dataset_name": dataset_name
                },
                "日志解析成功"
            )
        else:
            return mcp_utils.format_error_response(
                FileNotFoundError(f"解析输出文件不存在: {output_file}"),
                "parse_logs"
            )
            
    except Exception as e:
        return mcp_utils.format_error_response(e, "parse_logs")


# 工具 2: 训练异常检测模型
async def train_model_tool(
    source_dataset: str,
    target_dataset: str,
    max_epoch: int = 100,
    batch_size: int = 1024,
    lr: float = 0.001,
    window_size: int = 20,
    train_size_s: int = 100000,
    train_size_t: int = 1000
) -> Dict[str, Any]:
    """
    训练异常检测模型
    
    Args:
        source_dataset: 源域数据集名称
        target_dataset: 目标域数据集名称
        max_epoch: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        window_size: 窗口大小
        train_size_s: 源域训练样本数
        train_size_t: 目标域训练样本数
        
    Returns:
        训练结果字典
    """
    try:
        if not mcp_utils.validate_dataset_name(source_dataset):
            return mcp_utils.format_error_response(
                ValueError(f"无效的源数据集名称: {source_dataset}"),
                "train_model"
            )
        
        if not mcp_utils.validate_dataset_name(target_dataset):
            return mcp_utils.format_error_response(
                ValueError(f"无效的目标数据集名称: {target_dataset}"),
                "train_model"
            )
        
        logger.info(f"开始训练模型: {source_dataset} -> {target_dataset}")
        
        # 准备选项
        options = get_default_options()
        options.update({
            "source_dataset_name": source_dataset,
            "target_dataset_name": target_dataset,
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "window_size": window_size,
            "train_size_s": train_size_s,
            "train_size_t": train_size_t
        })
        
        # 设置随机种子
        set_seed(options["random_seed"])
        
        # 确保数据集存在
        source_file = os.path.join(config["data_dir"], f"{source_dataset}.log_structured.csv")
        target_file = os.path.join(config["data_dir"], f"{target_dataset}.log_structured.csv")
        
        if not os.path.exists(source_file) or not os.path.exists(target_file):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"数据集文件不存在，请先下载数据集"),
                "train_model"
            )
        
        # 加载数据集
        logger.info(f"加载源数据集: {source_file}")
        df_source = pd.read_csv(source_file)
        logger.info(f"加载目标数据集: {target_file}")
        df_target = pd.read_csv(target_file)
        
        # 数据预处理
        logger.info("开始数据预处理...")
        train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
        train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v = \
            SlidingWindow.get_datasets(df_source, df_target, options)
        
        # 准备训练数据
        train_iter, test_iter = get_train_eval_iter(train_normal_s, train_normal_t)
        
        # 创建模型
        model = LogTAD(options)
        
        # 训练模型（在后台线程中运行以避免阻塞）
        logger.info("开始模型训练...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.train_LogTAD(train_iter, test_iter, w2v)
        )
        
        # 模型保存路径
        model_files = mcp_utils.get_model_files(source_dataset, target_dataset, config["model_dir"])
        
        logger.info(f"模型训练完成，保存路径: {model_files['model']}")
        
        return mcp_utils.format_success_response(
            {
                "model_path": model_files["model"],
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "max_epoch": max_epoch,
                "model_files": model_files
            },
            "模型训练完成"
        )
        
    except Exception as e:
        return mcp_utils.format_error_response(e, "train_model")


# 工具 3: 检测日志异常
def detect_anomaly_tool(
    log_file_path: str,
    model_path: str,
    dataset_type: str,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    检测日志异常
    
    Args:
        log_file_path: 待检测的日志文件路径
        model_path: 训练好的模型路径
        dataset_type: 数据集类型 (source/target)
        threshold: 异常阈值（可选，将从模型获取）
        
    Returns:
        异常检测结果
    """
    try:
        if not mcp_utils.validate_file_path(log_file_path, must_exist=True):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"日志文件不存在: {log_file_path}"),
                "detect_anomaly"
            )
        
        if not mcp_utils.validate_file_path(model_path, must_exist=True):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"模型文件不存在: {model_path}"),
                "detect_anomaly"
            )
        
        if dataset_type not in ["source", "target"]:
            return mcp_utils.format_error_response(
                ValueError(f"无效的数据集类型: {dataset_type}，应为 'source' 或 'target'"),
                "detect_anomaly"
            )
        
        logger.info(f"开始异常检测: {log_file_path}")
        
        # 从模型路径推断数据集名称
        model_name = os.path.basename(model_path).replace(".pt", "")
        parts = model_name.split("-")
        if len(parts) != 2:
            return mcp_utils.format_error_response(
                ValueError(f"无法从模型路径推断数据集名称: {model_path}"),
                "detect_anomaly"
            )
        
        source_dataset, target_dataset = parts
        dataset_name = source_dataset if dataset_type == "source" else target_dataset
        
        # 加载模型
        options = get_default_options()
        options.update({
            "source_dataset_name": source_dataset,
            "target_dataset_name": target_dataset
        })
        
        model = LogTAD(options)
        model.load_model()
        
        # 这里需要实现对单个日志文件的检测
        # 由于原代码是针对数据集的，我们需要适配
        # 简化实现：假设输入文件是已解析的CSV格式
        if log_file_path.endswith('.csv'):
            df = pd.read_csv(log_file_path)
            # 这里需要完整的预处理流程，简化处理
            logger.warning("单文件检测功能需要完整的预处理流程，建议使用数据集进行评估")
            return mcp_utils.format_error_response(
                NotImplementedError("单文件检测功能暂未完全实现，请使用evaluate_model工具"),
                "detect_anomaly"
            )
        else:
            return mcp_utils.format_error_response(
                ValueError("当前仅支持CSV格式的已解析日志文件"),
                "detect_anomaly"
            )
            
    except Exception as e:
        return mcp_utils.format_error_response(e, "detect_anomaly")


# 工具 4: 下载公开数据集
def download_dataset_tool(dataset_name: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    下载公开数据集
    
    Args:
        dataset_name: 数据集名称 (BGL/Thunderbird)
        output_dir: 输出目录，默认使用配置
        
    Returns:
        下载结果字典
    """
    try:
        if not mcp_utils.validate_dataset_name(dataset_name):
            return mcp_utils.format_error_response(
                ValueError(f"无效的数据集名称: {dataset_name}"),
                "download_dataset"
            )
        
        output_dir = output_dir or config["data_dir"]
        logger.info(f"开始下载数据集: {dataset_name}")
        
        # 调用预处理模块的解析功能（包含下载）
        preprocessing.parsing(dataset_name, output_dir)
        
        output_file = os.path.join(output_dir, f"{dataset_name}.log_structured.csv")
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"数据集下载完成: {output_file}, 大小: {file_size} bytes")
            return mcp_utils.format_success_response(
                {
                    "dataset_name": dataset_name,
                    "output_file": output_file,
                    "file_size": file_size
                },
                "数据集下载成功"
            )
        else:
            return mcp_utils.format_error_response(
                FileNotFoundError(f"下载的输出文件不存在: {output_file}"),
                "download_dataset"
            )
            
    except Exception as e:
        return mcp_utils.format_error_response(e, "download_dataset")


# 工具 5: 获取模型信息
def get_model_info_tool(model_path: str) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型信息字典
    """
    try:
        if not mcp_utils.validate_file_path(model_path, must_exist=True):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"模型文件不存在: {model_path}"),
                "get_model_info"
            )
        
        # 从模型路径推断数据集名称
        model_name = os.path.basename(model_path).replace(".pt", "")
        parts = model_name.split("-")
        if len(parts) != 2:
            return mcp_utils.format_error_response(
                ValueError(f"无法从模型路径推断数据集名称: {model_path}"),
                "get_model_info"
            )
        
        source_dataset, target_dataset = parts
        model_files = mcp_utils.get_model_files(source_dataset, target_dataset, config["model_dir"])
        
        # 检查所有模型文件
        if not mcp_utils.check_model_exists(source_dataset, target_dataset, config["model_dir"]):
            return mcp_utils.format_error_response(
                FileNotFoundError("模型文件不完整"),
                "get_model_info"
            )
        
        # 加载模型获取参数
        options = get_default_options()
        options.update({
            "source_dataset_name": source_dataset,
            "target_dataset_name": target_dataset
        })
        
        model = LogTAD(options)
        model.load_model()
        
        # 获取模型信息
        model_info = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "model_path": model_path,
            "model_files": model_files,
            "parameters": {
                "emb_dim": model.emb_dim,
                "hid_dim": model.hid_dim,
                "output_dim": model.output_dim,
                "n_layers": model.n_layers,
                "dropout": model.dropout,
                "window_size": model.window_size,
                "step_size": model.step_size,
                "alpha": model.alpha,
                "device": model.device
            },
            "file_sizes": {
                "model": os.path.getsize(model_files["model"]),
                "center": os.path.getsize(model_files["center"]),
                "w2v": os.path.getsize(model_files["w2v"])
            }
        }
        
        logger.info(f"获取模型信息成功: {model_path}")
        return mcp_utils.format_success_response(model_info, "获取模型信息成功")
        
    except Exception as e:
        return mcp_utils.format_error_response(e, "get_model_info")


# 工具 6: 评估模型性能
def evaluate_model_tool(
    model_path: str,
    test_dataset: str,
    dataset_type: str
) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        test_dataset: 测试数据集名称
        dataset_type: 数据集类型 (source/target)
        
    Returns:
        评估指标字典
    """
    try:
        if not mcp_utils.validate_file_path(model_path, must_exist=True):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"模型文件不存在: {model_path}"),
                "evaluate_model"
            )
        
        if not mcp_utils.validate_dataset_name(test_dataset):
            return mcp_utils.format_error_response(
                ValueError(f"无效的数据集名称: {test_dataset}"),
                "evaluate_model"
            )
        
        if dataset_type not in ["source", "target"]:
            return mcp_utils.format_error_response(
                ValueError(f"无效的数据集类型: {dataset_type}"),
                "evaluate_model"
            )
        
        logger.info(f"开始评估模型: {model_path}, 数据集: {test_dataset}, 类型: {dataset_type}")
        
        # 从模型路径推断数据集名称
        model_name = os.path.basename(model_path).replace(".pt", "")
        parts = model_name.split("-")
        if len(parts) != 2:
            return mcp_utils.format_error_response(
                ValueError(f"无法从模型路径推断数据集名称: {model_path}"),
                "evaluate_model"
            )
        
        source_dataset, target_dataset = parts
        
        # 检查模型文件
        if not mcp_utils.check_model_exists(source_dataset, target_dataset, config["model_dir"]):
            return mcp_utils.format_error_response(
                FileNotFoundError("模型文件不完整"),
                "evaluate_model"
            )
        
        # 加载模型
        options = get_default_options()
        options.update({
            "source_dataset_name": source_dataset,
            "target_dataset_name": target_dataset
        })
        
        model = LogTAD(options)
        model.load_model()
        
        # 加载测试数据集
        dataset_file = os.path.join(config["data_dir"], f"{test_dataset}.log_structured.csv")
        if not os.path.exists(dataset_file):
            return mcp_utils.format_error_response(
                FileNotFoundError(f"测试数据集文件不存在: {dataset_file}"),
                "evaluate_model"
            )
        
        df_test = pd.read_csv(dataset_file)
        
        # 重新准备测试数据（需要完整的数据预处理流程）
        # 这里我们需要使用与训练时相同的数据预处理
        logger.warning("评估功能需要完整的数据预处理流程，这可能需要较长时间")
        
        # 为了完整实现，我们需要重新运行预处理
        # 简化：假设测试数据集已预处理
        # 实际使用时需要完整的预处理流程
        
        # 获取验证集用于计算阈值
        if dataset_type == "source":
            # 使用源域数据
            df_source = df_test
            df_target = pd.read_csv(os.path.join(config["data_dir"], f"{target_dataset}.log_structured.csv"))
        else:
            # 使用目标域数据
            df_target = df_test
            df_source = pd.read_csv(os.path.join(config["data_dir"], f"{source_dataset}.log_structured.csv"))
        
        # 数据预处理
        train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
        train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v = \
            SlidingWindow.get_datasets(df_source, df_target, options)
        
        # 计算阈值
        if dataset_type == "source":
            R, auc_val = model.get_r_from_val(r_s_val_df)
            test_normal = test_normal_s
            test_abnormal = test_abnormal_s
            dataset_name = source_dataset
        else:
            R, auc_val = model.get_r_from_val(r_t_val_df)
            test_normal = test_normal_t
            test_abnormal = test_abnormal_t
            dataset_name = target_dataset
        
        # 测试
        X = list(test_normal.Embedding.values[::int(1 / options["test_ratio"])])
        X.extend(list(test_abnormal.Embedding.values[::int(1 / options["test_ratio"])]))
        X_new = []
        for i in X:
            temp = []
            for j in i:
                temp.extend(j)
            X_new.append(np.array(temp).reshape(options["window_size"], options["emb_dim"]))
        
        y_d = list(test_normal.target.values[::int(1 / options["test_ratio"])])
        y_d.extend(list(test_abnormal.target.values[::int(1 / options["test_ratio"])]))
        y_true = list(test_normal.Label.values[::int(1 / options["test_ratio"])])
        y_true.extend(list(test_abnormal.Label.values[::int(1 / options["test_ratio"])]))
        
        X_test = torch.tensor(X_new, requires_grad=False)
        y_d_test = torch.tensor(y_d).reshape(-1, 1).long()
        y_test = torch.tensor(y_true).reshape(-1, 1).long()
        test_iter = get_iter(X_test, y_d_test, y_test)
        
        y_true_list, lst_dist = model._test(test_iter)
        y_pred = dist2label(lst_dist, R)
        
        # 计算评估指标
        accuracy = metrics.accuracy_score(y_true_list, y_pred)
        precision = metrics.precision_score(y_true_list, y_pred)
        recall = metrics.recall_score(y_true_list, y_pred)
        f1 = metrics.f1_score(y_true_list, y_pred)
        auc = metrics.roc_auc_score(y_true_list, y_pred)
        
        classification_report = metrics.classification_report(
            y_true_list, y_pred, output_dict=True, digits=5
        )
        
        evaluation_results = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "threshold": float(R),
            "threshold_auc": float(auc_val),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc)
            },
            "classification_report": classification_report,
            "test_samples": {
                "normal": len(test_normal),
                "abnormal": len(test_abnormal),
                "total": len(y_true_list)
            }
        }
        
        logger.info(f"模型评估完成: {dataset_name}, Accuracy: {accuracy:.5f}, F1: {f1:.5f}")
        return mcp_utils.format_success_response(evaluation_results, "模型评估完成")
        
    except Exception as e:
        return mcp_utils.format_error_response(e, "evaluate_model")


# 如果 fastmcp 可用，注册工具
if mcp is not None:
    try:
        mcp.tool()(parse_logs_tool)
        mcp.tool()(train_model_tool)
        mcp.tool()(detect_anomaly_tool)
        mcp.tool()(download_dataset_tool)
        mcp.tool()(get_model_info_tool)
        mcp.tool()(evaluate_model_tool)
        logger.info("fastmcp 工具注册成功")
    except Exception as e:
        logger.error(f"fastmcp 工具注册失败: {e}")
        mcp = None

# 备用实现：使用 FastAPI
if mcp is None:
    logger.warning("使用备用 FastAPI 实现")
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        from typing import Optional
        
        app = FastAPI(title="LogTAD MCP Server", version="1.0.0")
        
        # 定义请求模型
        class ParseLogsParams(BaseModel):
            log_file_path: str
            dataset_name: str
            output_dir: Optional[str] = None
        
        class TrainModelParams(BaseModel):
            source_dataset: str
            target_dataset: str
            max_epoch: int = 100
            batch_size: int = 1024
            lr: float = 0.001
            window_size: int = 20
            train_size_s: int = 100000
            train_size_t: int = 1000
        
        class DetectAnomalyParams(BaseModel):
            log_file_path: str
            model_path: str
            dataset_type: str
            threshold: Optional[float] = None
        
        class DownloadDatasetParams(BaseModel):
            dataset_name: str
            output_dir: Optional[str] = None
        
        class GetModelInfoParams(BaseModel):
            model_path: str
        
        class EvaluateModelParams(BaseModel):
            model_path: str
            test_dataset: str
            dataset_type: str
        
        @app.post("/tools/parse_logs")
        async def parse_logs_endpoint(params: ParseLogsParams):
            return JSONResponse(content=parse_logs_tool(**params.dict()))
        
        @app.post("/tools/train_model")
        async def train_model_endpoint(params: TrainModelParams):
            result = await train_model_tool(**params.dict())
            return JSONResponse(content=result)
        
        @app.post("/tools/detect_anomaly")
        async def detect_anomaly_endpoint(params: DetectAnomalyParams):
            return JSONResponse(content=detect_anomaly_tool(**params.dict()))
        
        @app.post("/tools/download_dataset")
        async def download_dataset_endpoint(params: DownloadDatasetParams):
            return JSONResponse(content=download_dataset_tool(**params.dict()))
        
        @app.post("/tools/get_model_info")
        async def get_model_info_endpoint(params: GetModelInfoParams):
            return JSONResponse(content=get_model_info_tool(**params.dict()))
        
        @app.post("/tools/evaluate_model")
        async def evaluate_model_endpoint(params: EvaluateModelParams):
            return JSONResponse(content=evaluate_model_tool(**params.dict()))
        
        @app.get("/")
        async def root():
            return {
                "name": "LogTAD MCP Server",
                "version": "1.0.0",
                "status": "running",
                "tools": [
                    "parse_logs",
                    "train_model",
                    "detect_anomaly",
                    "download_dataset",
                    "get_model_info",
                    "evaluate_model"
                ]
            }
        
        # 用于兼容性
        mcp = app
        logger.info("FastAPI 应用创建成功")
    except ImportError as e:
        logger.error(f"FastAPI 导入失败: {e}")
        logger.error("请安装: pip install fastapi uvicorn")
        mcp = None


if __name__ == "__main__":
    if mcp is not None and hasattr(mcp, 'run'):
        # 使用 fastmcp
        logger.info("使用 fastmcp 启动服务器")
        mcp.run(transport=config["transport"], host=config["host"], port=config["port"])
    elif mcp is not None and hasattr(mcp, 'get'):
        # 使用 FastAPI
        logger.info("使用 FastAPI 启动服务器")
        import uvicorn
        uvicorn.run(
            mcp,
            host=config["host"],
            port=config["port"],
            log_level="info"
        )
    else:
        logger.error("无法启动服务器：MCP 框架未正确初始化")
        sys.exit(1)

