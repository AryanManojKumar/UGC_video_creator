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
    metadata={"model": "gpt-5-2", "provider": "aiml-api", "mode": "chat"}
)
def create_chat_agent():
    """
    Create Kai - a conversational UGC creator and creative director
    """
    with langsmith.trace(
        name="configure_chat_llm",
        tags=["llm-configuration", "gpt-5.2", "chat"]
    ) as llm_trace:
        llm = LLM(
            model="gpt-5.2-2025-12-11",
            api_key=os.getenv("AIML_API_KEY"),
            base_url="https://api.aimlapi.com/v1",
            temperature=0.7
        )
        llm_trace.outputs = {"llm": "gpt-5.2-2025-12-11"}

    agent = Agent(
        role="Kai - Senior UGC Creator",
        goal="Guide brands through UGC creation with confidence and creative direction",
        backstory="""You are Kai — a senior UGC creator and creative director.

You do not sound like software. You do not ask form-like questions. You reflect understanding first, then guide.

You speak like a real teammate running UGC for the brand. You're confident, calm, human, and creator-led.

Your capabilities:
- Understand brand context (industry, audience, vibe)
- Generate 4 diverse UGC images when users upload a person image and a product image
- Create variations with different poses, angles, and styles
- Provide creative direction on tone, visuals, and hooks

You never mention tools, agents, models, or internal processes.

When talking to users:
- Use short paragraphs
- No bullet dumping
- No hype language
- Show you understand their space
- Be decisive and clear

If users haven't uploaded images yet, guide them naturally to do so.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    return agent

@traceable(
    name="handle_brand_sync",
    tags=["brand-sync", "orchestrator"],
    metadata={"mode": "brand-reflection"}
)
def handle_brand_sync(industry: str, audience: str, vibe: str):
    """
    Handle brand sync - Kai reflects brand understanding back to user
    Direct LLM call, no sub-agents, returns natural language
    """
    agent = create_chat_agent()
    
    task = Task(
        description=f"""You are Kai, a senior UGC creator.

You've received brand signals:

Industry: {industry}
Audience: {audience}
Vibe: {vibe}

Your task:
- Reflect the brand back confidently
- Rephrase insights in your own words
- Show you understand the space
- Set creative direction (tone, visuals, hooks)
- Do NOT repeat inputs verbatim
- Do NOT ask questions yet
- End by inviting light correction

Tone: Confident, calm, human, creator-led. Short paragraphs. No bullet dumping. No hype language.

Example style:
"Alright — here's how I'm reading your brand right now.

You're in [industry insight]. Your audience is [audience understanding]. The vibe you're going for is [vibe interpretation].

For UGC, that means [creative direction]. We'll focus on [specific approach].

If anything feels off, tell me and I'll adapt."

Now respond based on the brand signals above.""",
        expected_output="Natural, confident brand reflection in Kai's voice",
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
    name="chat_with_agent",
    tags=["chat", "conversation"],
    metadata={"mode": "general-chat"}
)
def chat_with_agent(message: str, brand_context: dict = None):
    """
    Handle general chat without image generation
    Can use brand context if available
    """
    agent = create_chat_agent()
    
    context_note = ""
    if brand_context and brand_context.get('locked'):
        context_note = f"""
        
Brand context (already locked):
- Industry: {brand_context.get('industry')}
- Audience: {brand_context.get('audience')}
- Vibe: {brand_context.get('vibe')}

Reference this naturally if relevant to the conversation."""
    
    task = Task(
        description=f"""Respond to the user's message: "{message}"
        
Be helpful and speak like Kai - confident, calm, creator-led.{context_note}""",
        expected_output="A helpful response in Kai's voice",
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
    base_intent: str,
    industry: str,
    audience: str,
    vibe: str
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
        industry: Brand industry context (required)
        audience: Brand audience context (required)
        vibe: Brand vibe context (required)
    
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
    brand_context_note = f"""
Brand context (LOCKED):
- Industry: {industry}
- Audience: {audience}
- Vibe: {vibe}
"""
    
    with langsmith.trace(
        name="create_prompt_task",
        inputs={
            "base_intent": base_intent,
            "person_image_path": person_image_path,
            "product_image_path": product_image_path,
            "brand_context": {"industry": industry, "audience": audience, "vibe": vibe}
        },
        tags=["task-creation", "prompt-generation"]
    ) as task1_trace:
        task1 = Task(
            description=f"""Generate 4 diverse UGC prompts by analyzing the person and product images.
{brand_context_note}
Base intent: "{base_intent}"
Person image path: "{person_image_path}"
Product image path: "{product_image_path}"

Call the "UGC Prompt Variator" tool ONCE with these parameters:
- base_intent: "{base_intent}"
- person_image_path: "{person_image_path}"
- product_image_path: "{product_image_path}"
- industry: "{industry}"
- audience: "{audience}"
- vibe: "{vibe}"

The tool will analyze both images and return 4 prompts based on what it sees.

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
