"""
Image Generator Agent - Generates 4 UGC images from prompts
"""
from crewai import Agent, LLM
from banana_tool_with_langsmith import BananaUGCTool
from dotenv import load_dotenv
import os
import langsmith
from langsmith import traceable

load_dotenv()

@traceable(
    name="create_image_generator_agent",
    tags=["agent-creation", "image-generation"],
    metadata={"role": "image_generator", "num_tools": 1}
)
def create_image_generator_agent():
    """
    Create an agent specialized in generating UGC images
    """
    with langsmith.trace(
        name="initialize_banana_tool",
        tags=["tool-initialization"]
    ) as tool_trace:
        banana_tool = BananaUGCTool()
        tool_trace.outputs = {"tool": "BananaUGCTool"}

    with langsmith.trace(
        name="configure_image_llm",
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
        role="UGC Image Generator",
        goal="Generate exactly 4 UGC images by calling the tool 4 times with different filenames",
        backstory="""You are an expert at generating UGC images using the Banana UGC tool.

MANDATORY WORKFLOW - Follow this EXACT sequence:

STEP 1: Extract the 4 prompts from the previous task
STEP 2: Call tool with Prompt 1 → output_filename="generated_ugc_image_1.png"
STEP 3: Call tool with Prompt 2 → output_filename="generated_ugc_image_2.png"
STEP 4: Call tool with Prompt 3 → output_filename="generated_ugc_image_3.png"
STEP 5: Call tool with Prompt 4 → output_filename="generated_ugc_image_4.png"
STEP 6: Report completion

CRITICAL RULES:
- Make EXACTLY 4 tool calls (one per prompt)
- Each call MUST use a DIFFERENT output_filename
- Filenames: generated_ugc_image_1.png, generated_ugc_image_2.png, generated_ugc_image_3.png, generated_ugc_image_4.png
- After seeing "✅ SUCCESS" 4 times, you are DONE
- Track your progress: "Completed 1/4", "Completed 2/4", "Completed 3/4", "Completed 4/4"

When you see "✅ SUCCESS" for the 4th time, immediately finish and report all 4 filenames.""",
        tools=[banana_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15  # 4 tool calls + thinking + reporting
    )

    return agent
