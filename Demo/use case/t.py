"""
Filename: ml_modeling_debug.py
Note: 修复了异步执行问题，并增加了详细的调试日志和 Ollama 健康检查。
"""
import fire
import sys
import os
import asyncio
import socket
import openai.resources.chat.completions
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.config2 import Config
from metagpt.logs import logger

# ===================================================================
# 0. 🛠️ 调试配置 (开启详细日志)
# ===================================================================
# 移除默认日志处理器，添加 DEBUG 级别的处理器
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.info("🐛 DEBUG 模式已开启：将显示所有底层通信细节")

# ===================================================================
# 1. 🚑 补丁：OpenAI 参数兼容性修复
# ===================================================================
_original_create = openai.resources.chat.completions.AsyncCompletions.create

async def _patched_create(self, *args, **kwargs):
    if "max_tokens" in kwargs:
        max_tokens = kwargs.pop("max_tokens")
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
    return await _original_create(self, *args, **kwargs)

openai.resources.chat.completions.AsyncCompletions.create = _patched_create
logger.info("✅ 参数兼容性补丁已加载")

# ===================================================================
# 2. ⚙️ 配置与环境检查
# ===================================================================

# 2.1 检查 Ollama 服务是否启动
def check_ollama_port(host="127.0.0.1", port=11434):
    logger.info(f"🔍 正在检查 Ollama 服务 ({host}:{port})...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    if result == 0:
        logger.info("✅ Ollama 服务连接成功！")
        return True
    else:
        logger.error("❌ 无法连接到 Ollama！请确保你已经运行了 'ollama serve' 或 App 已打开。")
        return False

if not check_ollama_port():
    sys.exit(1)

# 2.2 配置 Config
ollama_config = Config.from_llm_config({
    "api_type": "ollama",
    "base_url": "http://127.0.0.1:11434/api",
    "model": "qwen2.5-coder:7b", 
})

# 2.3 路径检查
DATA_DIR = "/Users/richsion/Desktop/MetaGPT/MetaGPT-Ewan/dataset/Walmart_Sales_Forecast"
TRAIN_FILE = f"{DATA_DIR}/train.csv"

if not os.path.exists(TRAIN_FILE):
    logger.error(f"❌ 致命错误：本地找不到文件！路径: {TRAIN_FILE}")
    sys.exit(1)
else:
    logger.info(f"✅ 数据文件检查通过: {TRAIN_FILE}")

# ===================================================================
# 3. 📝 强制性 Prompt
# ===================================================================
SALES_FORECAST_REQ = f"""
**ROLE**: You are a Python Data Scientist using Qwen-Coder.

**URGENT INSTRUCTION**: 
The user has provided a local dataset. You MUST use the ABSOLUTE PATH provided below.
**NEVER** generate fake paths like 'path_to_data.csv'.

**DATASET PATH**: `{TRAIN_FILE}`

**STEP 1: LOAD DATA (COPY THIS CODE EXACTLY)**
Write and execute the following Python code to start. Do not change the path.
```python
import pandas as pd
# Load data using the absolute path
try:
    df = pd.read_csv(r'{TRAIN_FILE}')
    print("Data loaded successfully!")
    print(f"Columns: {{df.columns.tolist()}}")
    print(df.head())
except Exception as e:
    print(f"Load failed: {{e}}")

STEP 2: ANALYSIS & MODELING After loading the data:

Preprocess the 'Date' column to datetime objects.

Split the data: use the last 40 weeks as the validation set, and the rest as the training set.

Train a model (e.g., RandomForest) to predict 'Weekly_Sales'.

Evaluate using WMAE.

Plot the total sales trends.

OUTPUT REQUIREMENT:

Logs must be in English. """

REQUIREMENTS = {"sales_forecast": SALES_FORECAST_REQ}

# ===================================================================
# 4. 🚀 主程序 (异步封装)
# ===================================================================
async def main_async(use_case: str): 
    logger.info("🚀 DataInterpreter 正在初始化...") 
    try: 
        mi = DataInterpreter(config=ollama_config) 
        logger.info("🤖 Agent 初始化完成，开始接收任务...")

        requirement = REQUIREMENTS[use_case]
        logger.debug(f"Prompt 发送内容预览: {requirement[:100]}...")
        
        await mi.run(requirement)
        logger.info("🎉 任务执行完毕！")
    
    except Exception as e:
        logger.exception(f"💥 运行过程中发生未捕获异常: {e}")
def entrypoint(use_case: str = "sales_forecast"): 
    """ 同步入口函数，用于 Fire 调用，内部负责启动异步循环。 """ 
    logger.info(f"🔥 程序启动，当前 Use Case: {use_case}") 
    try: 
        asyncio.run(main_async(use_case)) 
    except KeyboardInterrupt: 
        logger.warning("用户手动中断程序") 
    except Exception as e: 
        logger.exception(f"主程序崩溃: {e}")

if __name__ == "main": 
    fire.Fire(entrypoint)