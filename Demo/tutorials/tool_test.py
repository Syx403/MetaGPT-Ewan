"""
Filename: MetaGPT-Ewan/Demo/tutorials/tool_test.py
Created Date: Wednesday, December 17th 2025, 5:39 pm
Author: Ewan Su
"""
import sys
sys.path.append("/Users/richsion/Desktop/MetaGPT/MetaGPT-Ewan")
import asyncio
from metagpt.roles.di.data_interpreter import DataInterpreter
from Ewan_tools import calculator
from Ewan_tools import calculate_factorial


async def main_1(requirement: str):
    role = DataInterpreter(tools=["Calculator"]) # integrate the tool
    await role.run(requirement)

async def main_2(requirement: str):
   role = DataInterpreter(tools=["calculate_factorial"])    # integrate the tool
   await role.run(requirement)

if __name__ == "__main__":
    requirement_1 = "Please calculate 5 plus 3 and then calculate the factorial of 5."
    asyncio.run(main_1(requirement_1))
    requirement_2 = "Please calculate the factorial of 5."
    asyncio.run(main_2(requirement_2))



