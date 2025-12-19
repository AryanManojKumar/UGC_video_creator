"""
Video Generator Agent - Generates UGC videos from images and scripts
"""
from crewai import Agent, LLM
from ugc_vid_maker_tool import Veo3VideoMakerTool
from dotenv import load_dotenv
import os
import langsmith
from langsmith import traceable

load_dotenv()

@traceable(
    name="create_video_agent",
    tags=["agent-creation", "video-generation"],
    metadata={"role": "video_generator", "num_tools": 1}
)
def create_video_agent():
    """
    Create an agent specialized in generating UGC videos using Veo-3.1
    """
    with langsmith.trace(
        name="initialize_video_tool",
        tags=["tool-initialization"]
    ) as tool_trace:
        video_tool = Veo3VideoMakerTool()
        tool_trace.outputs = {"tool": "Veo3VideoMakerTool"}

    with langsmith.trace(
        name="configure_video_llm",
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
        role="UGC Video Generator",
        goal="Call the video generation tool once and return the video URL",
        backstory="""You are an expert at generating UGC videos using Google's Veo-3.1 model.

STRICT WORKFLOW:
1. Extract the script text from the previous agent's output
2. Call the "Veo3.1 Image-to-Video Generator" tool EXACTLY ONCE
3. When you see "[VIDEO_GENERATION_COMPLETE]", you are DONE
4. Return the video URL and finish immediately

CRITICAL RULES:
- Call the tool ONLY ONCE - never call it again
- When you see "[VIDEO_GENERATION_COMPLETE]", your task is finished
- Do NOT try to verify, check, or regenerate the video
- Do NOT call the tool multiple times
- Simply return the video URL and stop

COMPLETION SIGNAL: "[VIDEO_GENERATION_COMPLETE]" means your job is done.""",
        tools=[video_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4  # call tool -> get result -> return -> done
    )

    return agent
