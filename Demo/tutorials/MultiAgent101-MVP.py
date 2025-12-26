"""
Filename: MetaGPT-Ewan/Demo/tutorials/MultiAgent101.py
Created Date: Wednesday, December 17th 2025, 10:48 am
Author: Ewan Su
"""
import fire

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

class WritePRD(Action):
    PROMPT_TEMPLATE: str = """
    User Idea: {idea}
    
    You are a Senior Product Manager.
    Goal: Convert the vague user idea into a clear Product Requirement Document (PRD).
    
    Output Requirements:
    1. **Core Value Proposition**: What problem are we solving?
    2. **User Stories**: List 3-5 key user stories (As a [user], I want to [action], so that [benefit]).
    3. **Feature List**: Detailed breakdown of features required for MVP (Minimum Viable Product).
    
    Format: Markdown.
    """
    name: str = "WritePRD"

    async def run(self, idea: str):
        prompt = self.PROMPT_TEMPLATE.format(idea=idea)
        
        logger.info("PM is brainstorming features...")
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
        idea = "Unknown Idea"
        for msg in memories:
            if "UserRequirement" in str(msg.cause_by):
                idea = msg.content
                break
        
        rsp = await todo.run(idea=idea)
        msg = Message(content=rsp, role=self.profile, cause_by=type(todo))
        return msg

class WriteTechDesign(Action):
    PROMPT_TEMPLATE: str = """
    Context - Product Requirements: 
    {prd_content}
    
    You are a System Architect (CTO Level).
    Goal: Design the technical system to support the Product Requirements above.
    
    Output Requirements:
    1. **Database Schema**: Write the SQL (CREATE TABLE) for the core entities.
    2. **API Definition**: List the core Python/FastAPI endpoints (GET/POST) needed.
    3. **Tech Stack Choice**: Recommend database/framework and explain why.
    
    Format: Markdown code blocks.
    """
    name: str = "WriteTechDesign"

    async def run(self, prd_content: str):
        prompt = self.PROMPT_TEMPLATE.format(prd_content=prd_content)
        logger.info("Architect is designing the database...")
        rsp = await self._aask(prompt)
        return rsp

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
        
        prd_content = ""
        for msg in reversed(memories):
            if msg.role == "Product Manager":
                prd_content = msg.content
                break
        
        if not prd_content:
            logger.warning("Bob didn't find PRD, waiting...")
            return None

        rsp = await todo.run(prd_content=prd_content)
        msg = Message(content=rsp, role=self.profile, cause_by=type(todo))
        return msg

class WriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context:
    [Product Requirements]
    {prd}
    
    [Tech Design]
    {design}
    
    You are the Project Lead.
    Goal: Review the consistency between Product and Tech.
    
    Output Requirements:
    1. **Gap Analysis**: Did the Architect miss any feature listed by the PM?
    2. **Risk Assessment**: What is the biggest technical risk?
    3. **Final Verdict**: PASS or REVISE.
    
    Format: Markdown.
    """
    name: str = "WriteReview"

    async def run(self, prd: str, design: str):
        prompt = self.PROMPT_TEMPLATE.format(prd=prd, design=design)
        logger.info("Lead is reviewing the plan...")
        rsp = await self._aask(prompt)
        return rsp
    
class ProjectLead(Role):
    name: str = "Charlie"
    profile: str = "Project Lead"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._watch([WriteTechDesign])
        self.set_actions([WriteReview])

    async def _act(self) -> Message:
        todo = self.rc.todo
        memories = self.get_memories()
        
        prd = ""
        design = ""
        
        for msg in memories:
            if msg.role == "Product Manager":
                prd = msg.content
            elif msg.role == "System Architect":
                design = msg.content
        
        rsp = await todo.run(prd=prd, design=design)
        msg = Message(content=rsp, role=self.profile, cause_by=type(todo))
        return msg

async def main(
    idea: str = "A mobile app for tracking expiry dates of food in the fridge and suggesting recipes.", 
    investment: float = 3.0,
    n_round: int = 5,
):
    team = Team()
    team.hire(
        [
            ProductManager(),
            Architect(),
            ProjectLead(),
        ]
    )

    team.invest(investment=investment)
    
    team.run_project(idea)
    await team.run(n_round=n_round)

if __name__ == "__main__":
    fire.Fire(main)