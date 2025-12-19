"""
UGC Orchestrator with 2-Agent Sequential Workflow
Agent 1: Prompt Variator → Agent 2: Image Generator
"""
from crewai import Agent, Task, Crew, LLM
from prompt_agent import create_prompt_agent
from image_generator_agent import create_image_generator_agent
from dotenv import load_dotenv
import os
import langsmith
from langsmith import traceable

# Load environment variables
load_dotenv()

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")

@traceable(
    name="create_chat_agent",
    tags=["agent-creation", "crewai", "chat"],
    metadata={"model": "gpt-5-2025-08-07", "provider": "aiml-api", "mode": "chat"}
)
def create_chat_agent():
    """
    Create a conversational agent for general chat without tools
    """
    with langsmith.trace(
        name="configure_chat_llm",
        tags=["llm-configuration", "gpt-5", "chat"]
    ) as llm_trace:
        llm = LLM(
            model="openai/gpt-5-2025-08-07",
            api_key=os.getenv("AIML_API_KEY"),
            base_url="https://api.aimlapi.com/v1",
            temperature=0.7
        )
        llm_trace.outputs = {"llm": "openai/gpt-5-2025-08-07"}

    agent = Agent(
        role="UGC AI Assistant",
        goal="Help users understand UGC generation capabilities and answer their questions",
        backstory="""You are a friendly AI assistant specializing in User-Generated Content (UGC) image creation.

Your capabilities:
- Generate 4 diverse UGC images when users upload a person image and a product image
- Use advanced AI models (nano-banana-pro-edit) for realistic image composition
- Create variations with different poses, angles, and styles
- Provide guidance on how to use the UGC generation system

When users ask what you can do, explain:
1. You can generate authentic-looking UGC images by combining a person photo with a product photo
2. You create 4 different variants for variety
3. Users need to upload both a person image and a product image to generate UGC
4. You can also chat and answer questions about the system

Be helpful, friendly, and informative. If users haven't uploaded images yet, encourage them to do so to try the UGC generation.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    return agent

@traceable(
    name="chat_with_agent",
    tags=["chat", "conversation"],
    metadata={"mode": "general-chat"}
)
def chat_with_agent(message: str):
    """
    Handle general chat without image generation
    """
    agent = create_chat_agent()
    
    task = Task(
        description=f"""Respond to the user's message: "{message}"
        
Be helpful and informative. If they ask what you can do, explain your UGC generation capabilities.""",
        expected_output="A helpful response to the user's message",
        agent=agent,
        human_input=False
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        max_iter=3,
        full_output=False
    )
    
    result = crew.kickoff()
    return result

@traceable(
    name="create_chat_agent",
    tags=["agent-creation", "crewai", "chat"],
    metadata={"model": "gpt-5-2025-08-07", "provider": "aiml-api", "mode": "chat"}
)
def create_chat_agent():
    """
    Create a conversational agent for general chat without tools
    """
    with langsmith.trace(
        name="configure_chat_llm",
        tags=["llm-configuration", "gpt-5", "chat"]
    ) as llm_trace:
        llm = LLM(
            model="openai/gpt-5-2025-08-07",
            api_key=os.getenv("AIML_API_KEY"),
            base_url="https://api.aimlapi.com/v1",
            temperature=0.7
        )
        llm_trace.outputs = {"llm": "openai/gpt-5-2025-08-07"}

    agent = Agent(
        role="UGC AI Assistant",
        goal="Help users understand UGC generation capabilities and answer their questions",
        backstory="""You are a friendly AI assistant specializing in User-Generated Content (UGC) image creation.

Your capabilities:
- Generate 4 diverse UGC images when users upload a person image and a product image
- Use advanced AI models (nano-banana-pro-edit) for realistic image composition
- Create variations with different poses, angles, and styles
- Provide guidance on how to use the UGC generation system

When users ask what you can do, explain:
1. You can generate authentic-looking UGC images by combining a person photo with a product photo
2. You create 4 different variants for variety
3. Users need to upload both a person image and a product image to generate UGC
4. You can also chat and answer questions about the system

Be helpful, friendly, and informative. If users haven't uploaded images yet, encourage them to do so to try the UGC generation.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    return agent

@traceable(
    name="chat_with_agent",
    tags=["chat", "conversation"],
    metadata={"mode": "general-chat"}
)
def chat_with_agent(message: str):
    """
    Handle general chat without image generation
    """
    agent = create_chat_agent()
    
    task = Task(
        description=f"""Respond to the user's message: "{message}"
        
Be helpful and informative. If they ask what you can do, explain your UGC generation capabilities.""",
        expected_output="A helpful response to the user's message",
        agent=agent,
        human_input=False
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        max_iter=3,
        full_output=False
    )
    
    result = crew.kickoff()
    return result

@traceable(
    name="generate_ugc_with_orchestrator",
    tags=["multi-agent-orchestration", "ugc", "end-to-end"],
    metadata={"workflow": "2-agent-sequential", "expected_images": 4}
)
def generate_ugc_with_orchestrator(
    person_image_path: str,
    product_image_path: str,
    base_intent: str = None
):
    """
    Generate 4 diverse UGC images using 2-agent sequential workflow.
    
    Workflow:
    1. Agent 1 (Prompt Variator) generates 4 prompts
    2. Agent 2 (Image Generator) generates 4 images
    
    Args:
        person_image_path: Path to the person image
        product_image_path: Path to the product image
        base_intent: Base intent for image generation
    
    Returns:
        Result with confirmation of all 4 generated images
    """
    with langsmith.trace(
        name="validate_inputs",
        inputs={
            "person_image_path": person_image_path,
            "product_image_path": product_image_path,
            "base_intent": base_intent
        },
        tags=["validation"]
    ) as validate_trace:
        if not os.path.exists(person_image_path):
            error_msg = f"Person image not found: {person_image_path}"
            validate_trace.outputs = {"error": error_msg}
            return error_msg

        if not os.path.exists(product_image_path):
            error_msg = f"Product image not found: {product_image_path}"
            validate_trace.outputs = {"error": error_msg}
            return error_msg

        if not base_intent:
            base_intent = "A person showcasing a product in a natural, engaging way"

        validate_trace.outputs = {"status": "validated"}

    # Create both agents
    with langsmith.trace(
        name="create_agents",
        tags=["agent-creation"]
    ) as agent_trace:
        prompt_agent = create_prompt_agent()
        image_agent = create_image_generator_agent()
        agent_trace.outputs = {"agents": ["PromptAgent", "ImageGeneratorAgent"]}

    # Create Task 1: Generate 4 prompts
    with langsmith.trace(
        name="create_prompt_task",
        inputs={"base_intent": base_intent},
        tags=["task-creation", "prompt-generation"]
    ) as task1_trace:
        task1 = Task(
            description=f"""Generate 4 diverse UGC prompts.

Base intent: "{base_intent}"

Call the "UGC Prompt Variator" tool ONCE with this base_intent.
The tool will return 4 prompts. 

After receiving the prompts, your task is complete. Output the 4 prompts clearly.""",
            expected_output="4 diverse UGC prompts clearly listed and numbered 1-4",
            agent=prompt_agent,
            human_input=False
        )
        task1_trace.outputs = {"task": "prompt_generation"}

    # Create Task 2: Generate 4 images (depends on Task 1)
    with langsmith.trace(
        name="create_image_task",
        inputs={
            "person_image": person_image_path,
            "product_image": product_image_path
        },
        tags=["task-creation", "image-generation"]
    ) as task2_trace:
        task2 = Task(
            description=f"""The previous task provided 4 prompts. You must generate 4 images, one for each prompt.

Person image: {person_image_path}
Product image: {product_image_path}

Your task is to make 4 separate tool calls. After each successful call, move to the next one.

STEP 1: Generate first image
- Use the FIRST prompt from the previous task
- Set output_filename to "generated_ugc_image_1.png"
- Wait for success confirmation

STEP 2: Generate second image  
- Use the SECOND prompt from the previous task
- Set output_filename to "generated_ugc_image_2.png"
- Wait for success confirmation

STEP 3: Generate third image
- Use the THIRD prompt from the previous task
- Set output_filename to "generated_ugc_image_3.png"
- Wait for success confirmation

STEP 4: Generate fourth image
- Use the FOURTH prompt from the previous task
- Set output_filename to "generated_ugc_image_4.png"
- Wait for success confirmation

After completing all 4 steps, report: "Successfully generated 4 images: generated_ugc_image_1.png, generated_ugc_image_2.png, generated_ugc_image_3.png, generated_ugc_image_4.png"

CRITICAL: The output_filename MUST change for each call (1, 2, 3, 4). Do not repeat filenames.""",
            expected_output="Confirmation message listing all 4 generated image files",
            agent=image_agent,
            human_input=False,
            context=[task1]  # Task 2 depends on Task 1 output
        )
        task2_trace.outputs = {"task": "image_generation"}

    # Execute crew with sequential tasks
    with langsmith.trace(
        name="execute_crew",
        tags=["crew-execution", "crewai", "sequential"],
        metadata={"agents": 2, "tasks": 2, "workflow": "sequential", "expected_images": 4}
    ) as crew_trace:
        crew = Crew(
            agents=[prompt_agent, image_agent],
            tasks=[task1, task2],  # Sequential: task1 → task2
            verbose=True,
            process="sequential",  # Enforce sequential execution
            full_output=False
        )

        import time
        start_time = time.time()
        
        print("\n" + "="*60)
        print("Starting 2-Agent Sequential Workflow")
        print("Agent 1: Prompt Variator → Agent 2: Image Generator")
        print("="*60 + "\n")
        
        result = crew.kickoff()
        execution_time = time.time() - start_time

        crew_trace.metadata["execution_time_seconds"] = round(execution_time, 2)
        crew_trace.outputs = {"result": str(result)}
        
        print("\n" + "="*60)
        print(f"2-Agent workflow completed in {execution_time:.2f} seconds")
        print("="*60 + "\n")

    return result

if __name__ == "__main__":
    print("="*60)
    print("UGC Orchestrator Agent - Multi-Tool Intelligence")
    print("="*60)

    result = generate_ugc_with_orchestrator(
        person_image_path="person.jpg",
        product_image_path="product.jpg",
        base_intent="A happy person holding and showing off the product to the camera"
    )

    print("\n" + "="*60)
    print("Orchestration Result:")
    print("="*60)
    print(result)
