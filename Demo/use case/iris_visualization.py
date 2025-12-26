"""
Filename: MetaGPT-Ewan/Demo/use case/iris_visualization.py
Created Date: Wednesday, December 18th 2025, 10:15 pm
Author: Ewan Su
"""
import asyncio
from metagpt.logs import logger
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.utils.recovery_util import save_history
from metagpt.config2 import Config

ollama_config = Config.from_llm_config({
        "api_type": "ollama",
        "base_url": "http://127.0.0.1:11434/api",
        "model": "qwen2.5-coder:7b", 
    })

async def main(requirement: str = ""):

    di = DataInterpreter(config=ollama_config)
    rsp = await di.run(requirement)
    logger.info(rsp)
    save_history(role=di)


if __name__ == "__main__":

    requirement = "Run data analysis on sklearn Iris dataset, include a plot"
    asyncio.run(main(requirement))