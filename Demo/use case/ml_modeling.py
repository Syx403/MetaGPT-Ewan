"""
Filename: MetaGPT-Ewan/Demo/use case/ml_modeling.py
Created Date: Wednesday, December 18th 2025, 10:58 pm
Author: Ewan Su
"""
import fire

from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.config2 import Config
import openai.resources.chat.completions


_original_create = openai.resources.chat.completions.AsyncCompletions.create

async def _patched_create(self, *args, **kwargs):
    if "max_tokens" in kwargs:

        max_tokens = kwargs.pop("max_tokens")

        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
            print(f"🔧 [Patch] 已自动将 max_tokens={max_tokens} 替换为 max_completion_tokens")

    return await _original_create(self, *args, **kwargs)

openai.resources.chat.completions.AsyncCompletions.create = _patched_create
print("✅ OpenAI 参数兼容性补丁已应用！")

ollama_config = Config.from_llm_config({
        "api_type": "ollama",
        "base_url": "http://127.0.0.1:11434/api",
        "model": "qwen2.5-coder:7b", 
    })

gpt_config = Config.from_llm_config({
        "api_type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-proj-D",
        "model": "gpt-4o",
        "use_system_prompt": "false",
        "stream": "false",
    })
  

WINE_REQ = "Run data analysis on sklearn Wine recognition dataset, include a plot, and train a model to predict wine class (20% as validation), and show validation accuracy."

# DATA_DIR = "/Users/richsion/Desktop/MetaGPT/MetaGPT-Ewan/dataset/Walmart_Sales_Forecast"

# SALES_FORECAST_REQ = f"""Train a model to predict sales for each department in every store (split the last 40 weeks records as validation dataset, the others is train dataset), include plot total sales trends, print metric and plot scatter plots of
# groud truth and predictions on validation data. Dataset is {DATA_DIR}/train.csv, the metric is weighted mean absolute error (WMAE) for test data. Notice: *print* key variables to get more information for next task step.
# """

DATA_DIR = "/Users/richsion/Desktop/MetaGPT/MetaGPT-Ewan/dataset/Walmart_Sales_Forecast"

SALES_FORECAST_REQ = f"""
**ROLE**: You are a professional Data Scientist.

**GOAL**: Train a model to predict sales using the Walmart dataset.

**CRITICAL INSTRUCTION**: 
1. The dataset file path is EXACTLY: "{DATA_DIR}/train.csv"
2. When writing code, **YOU MUST USE THE EXACT PATH ABOVE**. 
3. **DO NOT** use placeholders like 'path_to_your_dataset.csv' or 'data.csv'. 
4. **DO NOT** use relative paths. Use the full absolute path provided.

**TASKS**:
1. Load the dataset from "{DATA_DIR}/train.csv" and print the first 5 rows and the column names.
2. Split the last 40 weeks records as validation dataset, the others is train dataset.
3. Plot total sales trends.
4. Train a model to predict sales for each department in every store.
5. Evaluate using weighted mean absolute error (WMAE) on the validation data.
6. Plot scatter plots of ground truth vs predictions.
"""

REQUIREMENTS = {"wine": WINE_REQ, "sales_forecast": SALES_FORECAST_REQ}


async def main(use_case: str = "wine"):
    mi = DataInterpreter(config=ollama_config)
    # mi = DataInterpreter()
    requirement = REQUIREMENTS[use_case]
    await mi.run(requirement)


if __name__ == "__main__":
    fire.Fire(main)