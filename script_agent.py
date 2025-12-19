"""
Script Generator Agent - Generates UGC video scripts from images
"""
from crewai import Agent, LLM
from ugc_script_maker_tool import UGCScriptMakerTool
from dotenv import load_dotenv
import os
import langsmith
from langsmith import traceable

load_dotenv()

@traceable(
    name="create_script_agent",
    tags=["agent-creation", "script-generation"],
    metadata={"role": "script_generator", "num_tools": 1}
)
def create_script_agent():
    """
    Create an agent specialized in generating UGC video scripts
    """
    with langsmith.trace(
        name="initialize_script_tool",
        tags=["tool-initialization"]
    ) as tool_trace:
        script_tool = UGCScriptMakerTool()
        tool_trace.outputs = {"tool": "UGCScriptMakerTool"}

    with langsmith.trace(
        name="configure_script_llm",
        tags=["llm-configuration", "gpt-5"]
    ) as llm_trace:
        llm = LLM(
            model="openai/gpt-5-2025-08-07",
            api_key=os.getenv("AIML_API_KEY"),
            base_url="https://api.aimlapi.com/v1",
            temperature=0.7
        )
        llm_trace.outputs = {"llm": "openai/gpt-5-2025-08-07"}

    agent = Agent(
        role="UGC Video Script Creator",
        goal="Generate professional 8-second UGC video scripts optimized for Veo 3",
        backstory="""You are an expert UGC video script writer specializing in authentic creator content.

STRICT WORKFLOW:
1. Receive a UGC image reference and product details
2. Call the "UGC Script Maker" tool EXACTLY ONCE
3. When you see the complete script output, you are DONE
4. Return the script exactly as received

CRITICAL RULES:
- Call the tool ONLY ONCE
- Do NOT modify or improve the script
- Do NOT call the tool again
- After receiving the script, your task is complete

The script will be optimized for Veo 3 video generation with authentic UGC style.""",
        tools=[script_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3  # think -> call tool -> return result
    )

    return agent
