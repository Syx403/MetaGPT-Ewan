"""
Filename: MetaGPT-Ewan/Demo/tutorials/ollama_test.py
Created Date: Wednesday, December 18th 2025, 2:48 am
Author: Ewan Su
"""
import fire
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
from metagpt.environment import Environment  
from metagpt.config2 import Config

ollama_config = Config.from_llm_config({
        "api_type": "ollama",
        "base_url": "http://127.0.0.1:11434/api",
        "model": "qwen2.5-coder:7b", 
    })

class WritePRD(Action):
    PROMPT_TEMPLATE: str = """
    User Idea: {idea}
    
    You are a Senior Product Manager.
    Goal: Convert the vague user idea into a clear Product Requirement Document (PRD).
    
    Output Requirements:
    1. **Core Value Proposition**: What problem are we solving?
    2. **User Stories**: List 3-5 key user stories.
    3. **Feature List**: Detailed breakdown of features for MVP.
    
    Format: Markdown.
    """
    name: str = "WritePRD"

    async def run(self, idea: str):
        prompt = self.PROMPT_TEMPLATE.format(idea=idea)
        logger.info("👩‍💼 PM (GPT-4o-mini) is brainstorming features...")
        rsp = await self._aask(prompt)
        return rsp

class WriteTechDesign(Action):
    PROMPT_TEMPLATE: str = """
    Context - Product Requirements: 
    {prd_content}
    
    You are a System Architect.
    Goal: Design the technical system.
    
    Output Requirements:
    1. **File Structure**: List the file names needed (e.g. main.py, game.py).
    2. **API/Class Definition**: Define the core classes and methods.
    
    Format: Markdown code blocks.
    """
    name: str = "WriteTechDesign"

    async def run(self, prd_content: str):
        prompt = self.PROMPT_TEMPLATE.format(prd_content=prd_content)
        logger.info("👷‍♂️ Architect (GPT-4o-mini) is designing the system...")
        rsp = await self._aask(prompt)
        return rsp

class WriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Context - Tech Design:
    {design_content}
    
    You are a Senior Python Engineer.
    Goal: Write the COMPLETE executable Python code for the Snake Game based on the design.
    
    Requirement:
    - Use 'pygame' library or standard 'curses' library (for CLI).
    - Provide a single executable file content if possible, or multiple blocks.
    - Ensure the code is bug-free and ready to run.
    
    Format: Python Code Block.
    """
    name: str = "WriteCode"

    async def run(self, design_content: str):
        logger.warning(f"👨‍💻 Engineer is running on Model: [{self.llm.model}] (Should be Qwen/Ollama)")
        prompt = self.PROMPT_TEMPLATE.format(design_content=design_content)
        logger.info("👨‍💻 Engineer (Local Qwen) is writing code... (Speed depends on your Mac)")
        rsp = await self._aask(prompt)
        return rsp

class WriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context:
    [Code Implementation]
    {code}
    
    You are the Project Lead.
    Goal: Review the code quality.
    
    Output Requirements:
    1. **Bugs**: Any obvious errors?
    2. **Score**: Rate from 1-10.
    3. **Verdict**: PASS or REVISE.
    
    Format: Markdown.
    """
    name: str = "WriteReview"

    async def run(self, code: str):
        prompt = self.PROMPT_TEMPLATE.format(code=code)
        logger.info("🕵️‍♂️ Lead (GPT-4o-mini) is reviewing the code...")
        rsp = await self._aask(prompt)
        return rsp

class ProductManager(Role):
    name: str = "Alice"
    profile: str = "Product Manager"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([WritePRD])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: Analyzing user requirement...")
        todo = self.rc.todo
        memories = self.get_memories()
        idea = next((msg.content for msg in memories if "UserRequirement" in str(msg.cause_by)), "Unknown Idea")
        
        rsp = await todo.run(idea=idea)
        return Message(content=rsp, role=self.profile, cause_by=type(todo))

class Architect(Role):
    name: str = "Bob"
    profile: str = "System Architect"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([WritePRD])
        self.set_actions([WriteTechDesign])

    async def _act(self) -> Message:
        todo = self.rc.todo
        memories = self.get_memories()
        prd_content = next((msg.content for msg in reversed(memories) if msg.role == "Product Manager"), None)
        
        if not prd_content: return None
        rsp = await todo.run(prd_content=prd_content)
        return Message(content=rsp, role=self.profile, cause_by=type(todo))

class Engineer(Role):
    name: str = "Alex"
    profile: str = "Engineer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([WriteTechDesign])
        self.set_actions([WriteCode])

    async def _act(self) -> Message:
        todo = self.rc.todo
        memories = self.get_memories()
        design_content = next((msg.content for msg in reversed(memories) if msg.role == "System Architect"), None)
        
        if not design_content: return None
        rsp = await todo.run(design_content=design_content)
        return Message(content=rsp, role=self.profile, cause_by=type(todo))

class ProjectLead(Role):
    name: str = "Charlie"
    profile: str = "Project Lead"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([WriteCode])
        self.set_actions([WriteReview])

    async def _act(self) -> Message:
        todo = self.rc.todo
        memories = self.get_memories()
        code = next((msg.content for msg in reversed(memories) if msg.role == "Engineer"), None)
        
        if not code: return None
        rsp = await todo.run(code=code)
        return Message(content=rsp, role=self.profile, cause_by=type(todo))


async def main(
    idea: str = "Write a command line snake game in a single file using python curses.", 
    investment: float = 3.0,
    n_round: int = 5,
):

    team = Team(env=Environment())
    team.hire(
        [
            ProductManager(), 
            Architect(),
            Engineer(config = ollama_config),
            ProjectLead(),
        ]
    )

    team.invest(investment=investment)
    team.run_project(idea)
    await team.run(n_round=n_round)

if __name__ == "__main__":
    fire.Fire(main)