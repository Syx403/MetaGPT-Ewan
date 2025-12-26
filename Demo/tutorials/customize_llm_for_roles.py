"""
Filename: MetaGPT-Ewan/Demo/tutorials/customize_llm_for_roles.py
Created Date: Wednesday, December 18th 2025, 3:35 pm
Author: Ewan Su
Comment: Priority of configuration is: Action config > Role config > Global config
"""
import asyncio
import sys 
from metagpt.logs import logger 
from metagpt.config2 import Config
from metagpt.roles import Role
from metagpt.actions import Action, UserRequirement
from metagpt.environment import Environment
from metagpt.team import Team

logger.remove()
logger.add(sys.stderr, level="INFO")

ollama_config = Config.from_llm_config({
        "api_type": "ollama",
        "base_url": "http://127.0.0.1:11434/api",
        "model": "qwen2.5-coder:7b", 
    })

# gpt5 = Config.default()
# gpt5.llm.model = "gpt-5-nano"

class DemocraticSay(Action):
    """Democratic exclusive action"""
    pass

class RepublicanSay(Action):
    """Republican exclusive action作"""
    pass

class VoteAction(Action):
    """Vote action"""
    pass

a1 = DemocraticSay(name="Say", instruction="Say your opinion with emotion and don't repeat it")
a2 = RepublicanSay(config=ollama_config, name="Say", instruction="Say your opinion with emotion and don't repeat it")
a3 = VoteAction(name="Vote", instruction="Vote for the candidate, and say why you vote for him/her")


A = Role(name="A", profile="Democratic candidate", goal="Win the election", actions=[a1], watch=[UserRequirement, a2])
B = Role(name="B", profile="Republican candidate", goal="Win the election", actions=[a2], watch=[a1], config=ollama_config)
C = Role(name="C", profile="Voter", goal="Vote for the candidate", actions=[a3], watch=[a2]) 

async def main():
    print("🚀 Program starting...")
    env = Environment(desc="US election live broadcast")
    team = Team(investment=5.0, env=env, roles=[A, B, C])
    
    await team.run(idea="Topic: climate change. Under 80 words per message.", send_to="A", n_round=3)

if __name__ == "__main__":
    asyncio.run(main())