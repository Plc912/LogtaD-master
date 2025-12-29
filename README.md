# LogTAD MCP

MCP工具制作者:Plc12

邮箱：3522236586@qq.com

GitHub:https://github.com/Plc912/LogtaD-master.git

## 简介

LogTAD MCP 服务器将 LogTAD 日志异常检测项目封装为 MCP (Model Context Protocol) 工具，提供标准化的 REST API 接口，支持通过 SSE (Server-Sent Events) 进行实时通信。

## 功能特性

- ✅ 日志解析：使用 Drain 算法解析日志文件
- ✅ 模型训练：训练领域自适应异常检测模型
- ✅ 异常检测：对新日志进行异常检测
- ✅ 数据集管理：下载和管理 BGL/Thunderbird 数据集
- ✅ 模型信息查询：获取训练好的模型详细信息
- ✅ 模型评估：评估模型性能指标

## 安装

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 MCP 额外依赖

```bash
pip install -r requirements_mcp.txt
```

### 3. 配置

编辑 `mcp_config.json` 文件以配置服务器参数：

```json
{
  "host": "127.0.0.1",
  "port": 4003,
  "transport": "sse",
  "model_dir": "./saved_model",
  "data_dir": "./Dataset",
  "log_dir": "./logs"
}
```

## 启动服务器

### 方式 1: 使用启动脚本（推荐）

```bash
python start_mcp_server.py
```

### 方式 2: 直接运行

```bash
python mcp_server.py
```

服务器将在 `http://127.0.0.1:4003` 启动。

## API 接口

### 1. 下载数据集 (download_dataset)

下载公开的日志数据集（BGL 或 Thunderbird）。

**请求示例：**

```python
import requests

response = requests.post(
    "http://127.0.0.1:4003/tools/download_dataset",
    json={
        "dataset_name": "BGL",
        "output_dir": "./Dataset"  # 可选
    }
)

result = response.json()
print(result)
```

**响应示例：**

```json
{
  "success": true,
  "message": "数据集下载成功",
  "data": {
    "dataset_name": "BGL",
    "output_file": "./Dataset/BGL.log_structured.csv",
    "file_size": 12345678
  }
}
```

### 2. 解析日志 (parse_logs)

解析日志文件（需要先下载原始数据集）。

**请求示例：**

```python
response = requests.post(
    "http://127.0.0.1:4003/tools/parse_logs",
    json={
        "log_file_path": "./BGL.log",
        "dataset_name": "BGL",
        "output_dir": "./Dataset"  # 可选
    }
)
```

### 3. 训练模型 (train_model)

训练领域自适应异常检测模型。

**请求示例：**

```python
response = requests.post(
    "http://127.0.0.1:4003/tools/train_model",
    json={
        "source_dataset": "Thunderbird",
        "target_dataset": "BGL",
        "max_epoch": 100,          # 可选，默认 100
        "batch_size": 1024,        # 可选，默认 1024
        "lr": 0.001,               # 可选，默认 0.001
        "window_size": 20,         # 可选，默认 20
        "train_size_s": 100000,    # 可选，默认 100000
        "train_size_t": 1000       # 可选，默认 1000
    }
)

result = response.json()
if result["success"]:
    print(f"模型保存路径: {result['data']['model_path']}")
```

**注意事项：**

- 训练过程可能需要 10-30 分钟
- 建议使用 GPU 加速（自动检测）
- 训练结果保存在 `./saved_model/` 目录

### 4. 获取模型信息 (get_model_info)

获取训练好的模型详细信息。

**请求示例：**

```python
response = requests.post(
    "http://127.0.0.1:4003/tools/get_model_info",
    json={
        "model_path": "./saved_model/Thunderbird-BGL.pt"
    }
)

result = response.json()
if result["success"]:
    model_info = result["data"]
    print(f"源数据集: {model_info['source_dataset']}")
    print(f"目标数据集: {model_info['target_dataset']}")
    print(f"模型参数: {model_info['parameters']}")
```

**响应示例：**

```json
{
  "success": true,
  "message": "获取模型信息成功",
  "data": {
    "source_dataset": "Thunderbird",
    "target_dataset": "BGL",
    "model_path": "./saved_model/Thunderbird-BGL.pt",
    "parameters": {
      "emb_dim": 300,
      "hid_dim": 128,
      "output_dim": 2,
      "n_layers": 2,
      "dropout": 0.3,
      "window_size": 20,
      "step_size": 4,
      "alpha": 0.1,
      "device": "cuda"
    },
    "file_sizes": {
      "model": 123456,
      "center": 1234,
      "w2v": 567890
    }
  }
}
```

### 5. 评估模型 (evaluate_model)

评估模型在测试集上的性能。

**请求示例：**

```python
response = requests.post(
    "http://127.0.0.1:4003/tools/evaluate_model",
    json={
        "model_path": "./saved_model/Thunderbird-BGL.pt",
        "test_dataset": "BGL",
        "dataset_type": "target"  # "source" 或 "target"
    }
)

result = response.json()
if result["success"]:
    metrics = result["data"]["metrics"]
    print(f"准确率: {metrics['accuracy']:.5f}")
    print(f"精确率: {metrics['precision']:.5f}")
    print(f"召回率: {metrics['recall']:.5f}")
    print(f"F1 分数: {metrics['f1_score']:.5f}")
    print(f"AUC: {metrics['auc']:.5f}")
```

**响应示例：**

```json
{
  "success": true,
  "message": "模型评估完成",
  "data": {
    "dataset_name": "BGL",
    "dataset_type": "target",
    "threshold": 2.345,
    "threshold_auc": 0.98765,
    "metrics": {
      "accuracy": 0.92345,
      "precision": 0.91234,
      "recall": 0.93456,
      "f1_score": 0.92340,
      "auc": 0.97654
    },
    "classification_report": {
      "0": {"precision": 0.92, "recall": 0.93, "f1-score": 0.925},
      "1": {"precision": 0.91, "recall": 0.90, "f1-score": 0.905}
    },
    "test_samples": {
      "normal": 5000,
      "abnormal": 500,
      "total": 5500
    }
  }
}
```

### 6. 检测异常 (detect_anomaly)

检测日志文件中的异常（功能正在完善中）。

**请求示例：**

```python
response = requests.post(
    "http://127.0.0.1:4003/tools/detect_anomaly",
    json={
        "log_file_path": "./test_logs.csv",
        "model_path": "./saved_model/Thunderbird-BGL.pt",
        "dataset_type": "target",
        "threshold": 2.5  # 可选
    }
)
```

## 完整工作流程示例

### 示例 1: 训练和评估模型

```python
import requests
import time

BASE_URL = "http://127.0.0.1:4003/tools"

# 1. 下载数据集
print("1. 下载源数据集...")
response = requests.post(
    f"{BASE_URL}/download_dataset",
    json={"dataset_name": "Thunderbird"}
)
print(response.json())

print("2. 下载目标数据集...")
response = requests.post(
    f"{BASE_URL}/download_dataset",
    json={"dataset_name": "BGL"}
)
print(response.json())

# 2. 训练模型
print("3. 开始训练模型（这可能需要10-30分钟）...")
response = requests.post(
    f"{BASE_URL}/train_model",
    json={
        "source_dataset": "Thunderbird",
        "target_dataset": "BGL",
        "max_epoch": 50,  # 减少轮数以加快速度
        "train_size_s": 50000,
        "train_size_t": 1000
    }
)
train_result = response.json()
if train_result["success"]:
    model_path = train_result["data"]["model_path"]
    print(f"模型训练完成: {model_path}")
else:
    print(f"训练失败: {train_result}")
    exit(1)

# 3. 获取模型信息
print("4. 获取模型信息...")
response = requests.post(
    f"{BASE_URL}/get_model_info",
    json={"model_path": model_path}
)
model_info = response.json()
print(f"模型信息: {model_info['data']['parameters']}")

# 4. 评估模型
print("5. 评估模型性能...")
response = requests.post(
    f"{BASE_URL}/evaluate_model",
    json={
        "model_path": model_path,
        "test_dataset": "BGL",
        "dataset_type": "target"
    }
)
eval_result = response.json()
if eval_result["success"]:
    metrics = eval_result["data"]["metrics"]
    print(f"评估结果:")
    print(f"  准确率: {metrics['accuracy']:.5f}")
    print(f"  精确率: {metrics['precision']:.5f}")
    print(f"  召回率: {metrics['recall']:.5f}")
    print(f"  F1 分数: {metrics['f1_score']:.5f}")
    print(f"  AUC: {metrics['auc']:.5f}")
```

### 示例 2: 使用 curl 调用

```bash
# 下载数据集
curl -X POST http://127.0.0.1:4003/tools/download_dataset \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "BGL"}'

# 训练模型
curl -X POST http://127.0.0.1:4003/tools/train_model \
  -H "Content-Type: application/json" \
  -d '{
    "source_dataset": "Thunderbird",
    "target_dataset": "BGL",
    "max_epoch": 50
  }'

# 获取模型信息
curl -X POST http://127.0.0.1:4003/tools/get_model_info \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./saved_model/Thunderbird-BGL.pt"}'
```

## 错误处理

所有 API 接口都返回统一的响应格式：

**成功响应：**

```json
{
  "success": true,
  "message": "操作成功",
  "data": { ... }
}
```

**错误响应：**

```json
{
  "success": false,
  "error": {
    "type": "ValueError",
    "message": "错误描述",
    "context": "操作上下文"
  }
}
```

## 日志

服务器日志保存在 `./logs/logtad_mcp.log` 文件中，同时也会输出到控制台。

## 性能优化建议

1. **GPU 加速**：服务器会自动检测并使用 GPU（如果可用）
2. **批次大小**：根据 GPU 内存调整 `batch_size`
3. **训练样本数**：可以减少 `train_size_s` 和 `train_size_t` 以加快训练速度（但可能影响性能）
4. **并发请求**：模型训练会占用较多资源，建议避免并发训练请求

## 故障排除

### 问题 1: 模块导入错误

**错误：** `ModuleNotFoundError: No module named 'utils'`

**解决：** 确保在项目根目录运行服务器，或检查 Python 路径配置。

### 问题 2: CUDA 不可用

**错误：** `CUDA is not available`

**解决：** 服务器会自动回退到 CPU，但训练会变慢。确保安装了 PyTorch 的 CUDA 版本。

### 问题 3: 数据集下载失败

**错误：** 网络超时或下载失败

**解决：** 检查网络连接，或手动下载数据集到 `./Dataset/` 目录。

### 问题 4: 内存不足

**错误：** `Out of Memory`

**解决：**

- 减少 `batch_size`
- 减少 `train_size_s` 和 `train_size_t`
- 使用 CPU 模式（设置 `device: "cpu"`）

注意：下载转化数据集以及解析数据集训练模型会花费大量的时间，时间长短也取决于数据的多少，请合理规划时间。
