"""
FastAPI Chat-based UGC Orchestrator with LangSmith Monitoring

Wraps the existing CrewAI UGC agent for continuous chat interaction
with comprehensive LangSmith tracing and monitoring
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import base64
import uuid
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# LangSmith imports
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
import langsmith

# Import UGC orchestrator agent (true multi-tool intelligence)
from ugc_orchestrator_agent import generate_ugc_with_orchestrator

# Load environment variables
load_dotenv()

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ugc-orchestrator"
# Make sure to set LANGCHAIN_API_KEY in your .env file

langsmith_client = Client()

app = FastAPI(title="UGC Orchestrator API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history and generated images
conversations = {}
generated_images = {}

class ChatRequest(BaseModel):
    message: str
    person_image_path: Optional[str] = None
    product_image_path: Optional[str] = None
    conversation_id: Optional[str] = None

class ScriptRequest(BaseModel):
    ugc_image_path: str
    product_name: str
    avatar_id: int = 1  # Avatar ID for voice mapping
    tone: Optional[str] = "energetic and authentic"
    platform: Optional[str] = "Instagram"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    steps: List[Dict]
    generated_images: Optional[List[str]] = None  # Changed to list for 4 images
    timestamp: str
    trace_url: Optional[str] = None  # Added for LangSmith trace URL

class ScriptResponse(BaseModel):
    conversation_id: str
    script: str
    ugc_image_path: str
    timestamp: str
    trace_url: Optional[str] = None

@app.get("/")
async def serve_frontend():
    """Serve the chatbox HTML frontend"""
    if os.path.exists("chatbox.html"):
        return FileResponse("chatbox.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Frontend not found"}

@app.get("/old")
async def serve_old_frontend():
    """Serve the old grid-based HTML frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Frontend not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "UGC Orchestrator API",
        "version": "1.0.0",
        "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true"
    }

@traceable(
    name="chat_ugc_endpoint",
    tags=["fastapi", "ugc-generation"],
    metadata={"endpoint": "/chat/ugc"}
)
async def chat_ugc(request: ChatRequest):
    """Main chat endpoint - handles agent execution and image generation"""
    from ugc_orchestrator_agent import chat_with_agent
    
    conversation_id = request.conversation_id or str(uuid.uuid4())

    if conversation_id not in conversations:
        conversations[conversation_id] = []

    conversations[conversation_id].append({
        "role": "user",
        "message": request.message,
        "timestamp": datetime.now().isoformat()
    })

    run_tree = langsmith.get_current_run_tree()
    trace_url = None

    try:
        # Check if user wants to generate images (both images uploaded)
        if request.person_image_path and request.product_image_path:
            # Build image metadata
            image_metadata = {
                "person_image_uploaded": True,
                "product_image_uploaded": True,
            }

            if os.path.exists(request.person_image_path):
                image_metadata["person_image_size"] = os.path.getsize(request.person_image_path)
            if os.path.exists(request.product_image_path):
                image_metadata["product_image_size"] = os.path.getsize(request.product_image_path)

            # Track uploaded images
            with langsmith.trace(
                name="process_uploaded_images",
                inputs={
                    "person_image": request.person_image_path,
                    "product_image": request.product_image_path
                },
                tags=["image-upload"],
                metadata=image_metadata
            ) as image_trace:
                image_trace.outputs = {"status": "images_processed"}

            # Use agent orchestrator to generate 4 images
            print(f"\n{'='*60}")
            print(f"Processing UGC generation: {conversation_id}")
            print(f"User message: {request.message}")
            print(f"{'='*60}\n")
            
            with langsmith.trace(
                name="agent_orchestration",
                inputs={"message": request.message, "conversation_id": conversation_id},
                tags=["agent-execution", "multi-tool"]
            ) as agent_trace:
                result = generate_ugc_with_orchestrator(
                    person_image_path=request.person_image_path,
                    product_image_path=request.product_image_path,
                    base_intent=request.message
                )
                agent_trace.outputs = {"result": str(result)}
            
            assistant_message = str(result)
        else:
            # No images or partial images - use chat agent
            print(f"\n{'='*60}")
            print(f"Processing chat: {conversation_id}")
            print(f"User message: {request.message}")
            print(f"{'='*60}\n")
            
            with langsmith.trace(
                name="chat_agent",
                inputs={"message": request.message, "conversation_id": conversation_id},
                tags=["agent-execution", "chat"]
            ) as chat_trace:
                result = chat_with_agent(request.message)
                chat_trace.outputs = {"result": str(result)}
            
            assistant_message = str(result)
        
        steps = []

        # Handle multiple generated images (4 images)
        generated_image_list = []
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n{'='*60}")
        print("Checking for generated images...")
        print(f"{'='*60}")
        
        for i in range(1, 5):  # Check for 4 images
            original_filename = f"generated_ugc_image_{i}.png"
            print(f"Looking for: {original_filename}")
            
            if os.path.exists(original_filename):
                # Rename with conversation ID
                new_filename = f"ugc_{conversation_id}_{timestamp_str}_{i}.png"
                os.rename(original_filename, new_filename)
                generated_image_list.append(new_filename)
                print(f"✅ Found and renamed to: {new_filename}")
                
                # Build metadata
                image_size = os.path.getsize(new_filename)
                image_save_metadata = {
                    "filename": new_filename,
                    "size_bytes": image_size,
                    "conversation_id": conversation_id,
                    "variant_index": i
                }
                
                # Create trace with metadata
                with langsmith.trace(
                    name=f"save_generated_image_{i}",
                    tags=["image-output", "ugc-generation", f"variant-{i}"],
                    metadata=image_save_metadata
                ) as save_trace:
                    save_trace.outputs = {"image_path": new_filename}
            else:
                print(f"❌ Not found: {original_filename}")
        
        print(f"\nTotal images found: {len(generated_image_list)}")
        print(f"{'='*60}\n")
        
        # Store all generated images for this conversation
        if generated_image_list:
            generated_images[conversation_id] = generated_image_list
            
            # Log feedback
            try:
                if run_tree and run_tree.id:
                    langsmith_client.create_feedback(
                        run_tree.id,
                        key="images_generated",
                        score=1.0,
                        comment=f"{len(generated_image_list)} images generated: {', '.join(generated_image_list)}"
                    )
            except Exception as e:
                print(f"Failed to log feedback: {e}")

        steps.append({
            "type": "agent_thinking",
            "description": "GPT-5 agent analyzed the request",
            "timestamp": datetime.now().isoformat()
        })

        if "Success" in assistant_message and "generated_ugc_image" in assistant_message:
            steps.append({
                "type": "tool_call",
                "tool": "Multi Banana UGC Image Generator",
                "description": f"Generated {len(generated_image_list)} diverse UGC images using nano-banana-pro-edit model",
                "timestamp": datetime.now().isoformat()
            })

        conversations[conversation_id].append({
            "role": "assistant",
            "message": assistant_message,
            "timestamp": datetime.now().isoformat()
        })

        # Get trace URL
        if run_tree and run_tree.id:
            try:
                tenant_id = langsmith_client._get_tenant_id()
                project_name = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")
                trace_url = f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project_name}/r/{run_tree.id}"
            except Exception as e:
                print(f"Could not generate trace URL: {e}")
                trace_url = None

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            steps=steps,
            generated_images=generated_image_list if generated_image_list else None,
            timestamp=datetime.now().isoformat(),
            trace_url=trace_url
        )

    except Exception as e:
        print(f"Error in chat_ugc: {str(e)}")
        if run_tree and run_tree.id:
            langsmith_client.create_feedback(
                run_tree.id,
                key="error",
                score=0.0,
                comment=f"Error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@traceable(name="chat_ugc_upload_endpoint", tags=["fastapi", "file-upload"])
async def chat_ugc_with_upload(
    message: str = Form(...),
    person_image: Optional[UploadFile] = File(None),
    product_image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    """Upload wrapper - just saves files and calls chat_ugc()"""
    person_image_path = None
    product_image_path = None

    # Prepare metadata
    upload_metadata = {}

    if person_image:
        person_image_path = f"temp_person_{uuid.uuid4()}.jpg"
        content = await person_image.read()
        with open(person_image_path, "wb") as f:
            f.write(content)
        upload_metadata["person_image"] = {
            "filename": person_image.filename,
            "size": len(content),
            "path": person_image_path
        }

    if product_image:
        product_image_path = f"temp_product_{uuid.uuid4()}.jpg"
        content = await product_image.read()
        with open(product_image_path, "wb") as f:
            f.write(content)
        upload_metadata["product_image"] = {
            "filename": product_image.filename,
            "size": len(content),
            "path": product_image_path
        }

    # Log upload
    with langsmith.trace(
        name="save_uploaded_files", 
        tags=["file-upload"],
        metadata=upload_metadata
    ) as upload_trace:
        upload_trace.outputs = {"status": "files_saved"}

    # Create request
    request = ChatRequest(
        message=message,
        person_image_path=person_image_path,
        product_image_path=product_image_path,
        conversation_id=conversation_id
    )

    # ✅ JUST CALL chat_ugc() - it handles everything else
    response = await chat_ugc(request)

    # Clean up temp files
    if person_image_path and os.path.exists(person_image_path):
        os.remove(person_image_path)
    if product_image_path and os.path.exists(product_image_path):
        os.remove(product_image_path)

    # ✅ NO IMAGE HANDLING HERE - just return the response
    return response



@app.post("/chat/ugc/upload", response_model=ChatResponse)
async def chat_ugc_upload_endpoint(
    message: str = Form(...),
    person_image: Optional[UploadFile] = File(None),
    product_image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    """Wrapper endpoint"""
    return await chat_ugc_with_upload(message, person_image, product_image, conversation_id)

@app.post("/chat/ugc/upload/simple")
async def chat_ugc_upload_simple(
    message: str = Form(...),
    person_image: Optional[UploadFile] = File(None),
    product_image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    """Simple endpoint - just run and return results"""
    return await chat_ugc_with_upload(message, person_image, product_image, conversation_id)

@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    Retrieve a generated image by filename
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filename, media_type="image/png")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Retrieve a generated audio file by filename
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Audio not found")

    return FileResponse(filename, media_type="audio/mpeg")

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "generated_image": generated_images.get(conversation_id)
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    if conversation_id in conversations:
        del conversations[conversation_id]

    if conversation_id in generated_images:
        image_file = generated_images[conversation_id]
        if os.path.exists(image_file):
            os.remove(image_file)
        del generated_images[conversation_id]

    return {"status": "deleted", "conversation_id": conversation_id}

class ScriptVideoResponse(BaseModel):
    conversation_id: str
    script: str
    dialogue: Optional[str] = None
    audio_file: Optional[str] = None
    video_url: Optional[str] = None
    ugc_image_path: str
    avatar_id: int
    voice_used: Optional[str] = None
    timestamp: str
    trace_url: Optional[str] = None

@app.post("/chat/ugc/script", response_model=ScriptVideoResponse)
@traceable(
    name="generate_script_audio_video_endpoint",
    tags=["fastapi", "script-generation", "audio-generation", "video-generation"],
    metadata={"endpoint": "/chat/ugc/script", "workflow": "sequential"}
)
async def generate_script_and_video(request: ScriptRequest):
    """
    Generate UGC script, audio, and video (3-agent sequential workflow: script → audio → video)
    """
    from crewai import Task, Crew
    from script_agent import create_script_agent
    from audio_agent import create_audio_agent
    from video_agent import create_video_agent
    from ugc_audio_maker_tool import AVATAR_VOICE_MAP
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user request to conversation
    conversations[conversation_id].append({
        "role": "user",
        "message": f"Generate script, audio, and video for {request.ugc_image_path}",
        "timestamp": datetime.now().isoformat()
    })
    
    run_tree = langsmith.get_current_run_tree()
    trace_url = None
    
    try:
        # Validate image exists
        if not os.path.exists(request.ugc_image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.ugc_image_path}")
        
        # Get voice name from avatar_id
        voice_name = AVATAR_VOICE_MAP.get(request.avatar_id, "Harry")
        
        print(f"\n{'='*60}")
        print(f"Starting Sequential Workflow: Script → Audio → Video")
        print(f"Image: {request.ugc_image_path}")
        print(f"Product: {request.product_name}")
        print(f"Avatar ID: {request.avatar_id} (Voice: {voice_name})")
        print(f"Tone: {request.tone}")
        print(f"Platform: {request.platform}")
        print(f"{'='*60}\n")
        
        import time
        start_time = time.time()
        
        # ============================================================
        # STEP 1: Generate Script (Dialogue + Video Script)
        # ============================================================
        print("\n" + "="*60)
        print("STEP 1/3: Generating Script...")
        print("="*60)
        
        with langsmith.trace(
            name="step1_generate_script",
            tags=["script-generation"]
        ) as step1_trace:
            script_agent = create_script_agent()
            
            task1 = Task(
                description=f"""Generate an 8-second UGC video script with dialogue.

Call the "UGC Script Maker" tool with these parameters:
- ugc_image_reference: {request.ugc_image_path}
- product_name: {request.product_name}
- tone: {request.tone}
- platform: {request.platform}

Return the complete output with both dialogue and video script.""",
                expected_output="Complete output with UGC dialogue and video script sections",
                agent=script_agent,
                human_input=False
            )
            
            crew1 = Crew(
                agents=[script_agent],
                tasks=[task1],
                verbose=True,
                process="sequential"
            )
            
            script_result = crew1.kickoff()
            script_result_str = str(script_result)
            
            step1_trace.outputs = {"result": script_result_str[:500]}
        
        # Parse script result to extract dialogue and video script
        dialogue = ""
        video_script = ""
        
        if "=== UGC DIALOGUE ===" in script_result_str:
            parts = script_result_str.split("=== UGC DIALOGUE ===")
            if len(parts) > 1:
                dialogue_section = parts[1].split("=== VIDEO SCRIPT ===")[0].strip()
                dialogue = dialogue_section
        
        if "=== VIDEO SCRIPT ===" in script_result_str:
            parts = script_result_str.split("=== VIDEO SCRIPT ===")
            if len(parts) > 1:
                video_script = parts[1].strip()
        
        print(f"\n✅ Script Generated!")
        print(f"Dialogue: {dialogue[:80]}..." if len(dialogue) > 80 else f"Dialogue: {dialogue}")
        print(f"Video Script Length: {len(video_script)} characters")
        
        # ============================================================
        # STEP 2: Generate Audio from Dialogue
        # ============================================================
        print("\n" + "="*60)
        print("STEP 2/3: Generating Audio...")
        print("="*60)
        
        audio_filename = f"ugc_audio_{conversation_id}.mp3"
        audio_file = None
        
        with langsmith.trace(
            name="step2_generate_audio",
            tags=["audio-generation"],
            inputs={"dialogue": dialogue, "avatar_id": request.avatar_id}
        ) as step2_trace:
            audio_agent = create_audio_agent()
            
            task2 = Task(
                description=f"""Generate voice audio for this dialogue:

DIALOGUE TEXT:
{dialogue}

Call the "UGC Audio Generator" tool with:
- dialogue_text: {dialogue}
- avatar_id: {request.avatar_id}
- output_filename: {audio_filename}

Return the audio filename when complete.""",
                expected_output="Audio filename confirmation",
                agent=audio_agent,
                human_input=False
            )
            
            crew2 = Crew(
                agents=[audio_agent],
                tasks=[task2],
                verbose=True,
                process="sequential"
            )
            
            audio_result = crew2.kickoff()
            audio_result_str = str(audio_result)
            
            step2_trace.outputs = {"result": audio_result_str}
            
            # Check if audio was generated
            if "[AUDIO_GENERATION_COMPLETE]" in audio_result_str and audio_filename in audio_result_str:
                audio_file = audio_filename
                print(f"\n✅ Audio Generated: {audio_file}")
            else:
                print(f"\n⚠️ Audio generation may have failed")
        
        # ============================================================
        # STEP 3: Generate Video from Video Script
        # ============================================================
        print("\n" + "="*60)
        print("STEP 3/3: Generating Video...")
        print("="*60)
        
        video_url = None
        
        with langsmith.trace(
            name="step3_generate_video",
            tags=["video-generation"],
            inputs={"video_script": video_script[:200]}
        ) as step3_trace:
            video_agent = create_video_agent()
            
            task3 = Task(
                description=f"""Generate a UGC video using Veo-3.1.

VIDEO SCRIPT:
{video_script}

Call the "Veo3.1 Image-to-Video Generator" tool with:
- image_reference: {request.ugc_image_path}
- script_text: {video_script}
- duration_seconds: 8

Wait for the video generation to complete and return the video URL.""",
                expected_output="Video URL from Veo-3.1 generation",
                agent=video_agent,
                human_input=False
            )
            
            crew3 = Crew(
                agents=[video_agent],
                tasks=[task3],
                verbose=True,
                process="sequential"
            )
            
            video_result = crew3.kickoff()
            video_result_str = str(video_result)
            
            step3_trace.outputs = {"result": video_result_str[:500]}
            
            # Extract video URL
            if "Video generated successfully:" in video_result_str or "✅ Video generated successfully:" in video_result_str:
                if "✅ Video generated successfully:" in video_result_str:
                    parts = video_result_str.split("✅ Video generated successfully:")
                else:
                    parts = video_result_str.split("Video generated successfully:")
                
                if len(parts) > 1:
                    url_part = parts[1].strip()
                    url_part = url_part.split("[VIDEO_GENERATION_COMPLETE]")[0].strip()
                    video_url = url_part.split()[0] if url_part else None
                    print(f"\n✅ Video Generated: {video_url}")
            else:
                print(f"\n⚠️ Video generation may have failed")
        
        execution_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"📊 Workflow Complete in {execution_time:.2f} seconds")
        print(f"{'='*60}")
        print(f"Dialogue: {dialogue[:50]}..." if len(dialogue) > 50 else f"Dialogue: {dialogue}")
        print(f"Script Length: {len(video_script)} characters")
        print(f"Audio File: {audio_file if audio_file else 'Not generated'}")
        print(f"Video URL: {video_url if video_url else 'Not generated'}")
        print(f"Voice Used: {voice_name} (Avatar ID: {request.avatar_id})")
        print(f"{'='*60}\n")
        
        # Add to conversation
        conversations[conversation_id].append({
            "role": "assistant",
            "message": f"Generated script, audio, and video for {request.ugc_image_path}",
            "script": video_script,
            "dialogue": dialogue,
            "audio_file": audio_file,
            "video_url": video_url,
            "avatar_id": request.avatar_id,
            "voice_used": voice_name,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get trace URL
        if run_tree and run_tree.id:
            try:
                tenant_id = langsmith_client._get_tenant_id()
                project_name = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")
                trace_url = f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project_name}/r/{run_tree.id}"
            except Exception as e:
                print(f"Could not generate trace URL: {e}")
                trace_url = None
        
        return ScriptVideoResponse(
            conversation_id=conversation_id,
            script=video_script,
            dialogue=dialogue,
            audio_file=audio_file,
            video_url=video_url,
            ugc_image_path=request.ugc_image_path,
            avatar_id=request.avatar_id,
            voice_used=voice_name,
            timestamp=datetime.now().isoformat(),
            trace_url=trace_url
        )
        
    except Exception as e:
        print(f"Error generating script/audio/video: {str(e)}")
        if run_tree and run_tree.id:
            langsmith_client.create_feedback(
                run_tree.id,
                key="error",
                score=0.0,
                comment=f"Error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=f"Error generating script/audio/video: {str(e)}")

class LipsyncVideoRequest(BaseModel):
    ugc_image_path: str
    product_name: str
    avatar_id: int = 1
    tone: Optional[str] = "energetic and authentic"
    platform: Optional[str] = "Instagram"
    conversation_id: Optional[str] = None

class LipsyncVideoResponse(BaseModel):
    conversation_id: str
    script: str
    dialogue: str
    audio_file: str
    video_url: str
    lipsynced_video_url: str
    ugc_image_path: str
    avatar_id: int
    voice_used: str
    timestamp: str
    trace_url: Optional[str] = None

@app.post("/chat/ugc/generate-video", response_model=LipsyncVideoResponse)
@traceable(
    name="generate_complete_video_endpoint",
    tags=["fastapi", "complete-workflow", "lipsync"],
    metadata={"endpoint": "/chat/ugc/generate-video", "workflow": "script→audio→video→lipsync"}
)
async def generate_complete_video(request: LipsyncVideoRequest):
    """
    Complete 3-agent sequential workflow: Script → Audio+Video → Lipsync
    Returns final lipsynced video ready for use
    """
    from crewai import Task, Crew
    from script_agent import create_script_agent
    from lipsync_agent import create_lipsync_agent
    from ugc_audio_maker_tool import AVATAR_VOICE_MAP
    import requests
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        "role": "user",
        "message": f"Generate complete lipsynced video for {request.ugc_image_path}",
        "timestamp": datetime.now().isoformat()
    })
    
    run_tree = langsmith.get_current_run_tree()
    trace_url = None
    
    try:
        if not os.path.exists(request.ugc_image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.ugc_image_path}")
        
        voice_name = AVATAR_VOICE_MAP.get(request.avatar_id, "Harry")
        
        print(f"\n{'='*60}")
        print(f"Starting Complete Workflow: Script → Audio+Video → Lipsync")
        print(f"Image: {request.ugc_image_path}")
        print(f"Product: {request.product_name}")
        print(f"Avatar ID: {request.avatar_id} (Voice: {voice_name})")
        print(f"{'='*60}\n")
        
        import time
        start_time = time.time()
        
        # ============================================================
        # STEP 1: Generate Script
        # ============================================================
        print("\n" + "="*60)
        print("STEP 1/3: Generating Script...")
        print("="*60)
        
        with langsmith.trace(name="step1_script", tags=["script"]) as step1_trace:
            script_agent = create_script_agent()
            script_filename = f"script_{conversation_id}.txt"
            
            task1 = Task(
                description=f"""Use the UGC Script Maker tool now.

Tool: UGC Script Maker
Parameters:
  ugc_image_reference: {request.ugc_image_path}
  product_name: {request.product_name}
  tone: {request.tone}
  platform: {request.platform}
  output_filename: {script_filename}

Call the tool with these parameters.""",
                expected_output="Tool response with saved filename",
                agent=script_agent,
                human_input=False
            )
            
            crew1 = Crew(agents=[script_agent], tasks=[task1], verbose=True, process="sequential")
            script_result = crew1.kickoff()
            script_result_str = str(script_result)
            step1_trace.outputs = {"result": script_result_str[:500]}
        
        # Read script file
        if not os.path.exists(script_filename):
            raise HTTPException(status_code=500, detail="Script file not created")
        
        with open(script_filename, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        dialogue = ""
        video_script = ""
        
        if "=== UGC DIALOGUE ===" in script_content:
            parts = script_content.split("=== UGC DIALOGUE ===")
            if len(parts) > 1:
                dialogue = parts[1].split("=== VIDEO SCRIPT ===")[0].strip()
        
        if "=== VIDEO SCRIPT ===" in script_content:
            parts = script_content.split("=== VIDEO SCRIPT ===")
            if len(parts) > 1:
                video_script = parts[1].strip()
        
        print(f"✅ Script Generated: {script_filename}")
        print(f"Dialogue: {dialogue[:80]}...")
        
        # ============================================================
        # STEP 2: Generate Audio + Video (Combined)
        # ============================================================
        print("\n" + "="*60)
        print("STEP 2/3: Generating Audio + Video...")
        print("="*60)
        
        audio_filename = f"audio_{conversation_id}.mp3"
        video_url = None
        
        with langsmith.trace(name="step2_audio_video", tags=["audio-video"]) as step2_trace:
            from audio_video_agent import create_audio_video_agent
            audio_video_agent = create_audio_video_agent()
            
            task2 = Task(
                description=f"""Generate audio and video from the script below.

DIALOGUE (for audio):
{dialogue}

VIDEO SCRIPT (for video):
{video_script}

IMAGE REFERENCE: {request.ugc_image_path}
AVATAR ID: {request.avatar_id}
AUDIO OUTPUT: {audio_filename}

Tasks:
1. Call UGC Audio Generator with dialogue_text="{dialogue}", avatar_id={request.avatar_id}, output_filename="{audio_filename}"
2. Call Veo3.1 Image-to-Video Generator with image_reference="{request.ugc_image_path}", script_text=<video script above>, duration_seconds=8
3. Return audio filename and video URL""",
                expected_output="Audio filename and video URL",
                agent=audio_video_agent,
                human_input=False
            )
            
            crew2 = Crew(agents=[audio_video_agent], tasks=[task2], verbose=True, process="sequential")
            audio_video_result = crew2.kickoff()
            audio_video_result_str = str(audio_video_result)
            step2_trace.outputs = {"result": audio_video_result_str[:500]}
        
        # Verify audio file was created
        if not os.path.exists(audio_filename):
            raise HTTPException(status_code=500, detail="Audio file not created")
        
        print(f"✅ Audio Generated: {audio_filename}")
        
        # Extract video URL from result
        video_url = None
        
        # Try multiple patterns to extract video URL
        if "Video URL:" in audio_video_result_str:
            # Pattern: "Video URL:\n<url>"
            parts = audio_video_result_str.split("Video URL:")
            if len(parts) > 1:
                url_part = parts[1].strip()
                # Extract URL (first line after "Video URL:")
                lines = url_part.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('http'):
                        video_url = line.split()[0]
                        break
        
        if not video_url and ("Video generated successfully:" in audio_video_result_str or "✅ Video generated successfully:" in audio_video_result_str):
            # Fallback pattern: "Video generated successfully: <url>"
            if "✅ Video generated successfully:" in audio_video_result_str:
                parts = audio_video_result_str.split("✅ Video generated successfully:")
            else:
                parts = audio_video_result_str.split("Video generated successfully:")
            
            if len(parts) > 1:
                url_part = parts[1].strip()
                url_part = url_part.split("[VIDEO_GENERATION_COMPLETE]")[0].strip()
                video_url = url_part.split()[0] if url_part else None
        
        if not video_url:
            raise HTTPException(status_code=500, detail="Video URL not generated")
        
        print(f"✅ Video Generated: {video_url}")
        
        # ============================================================
        # STEP 3: Upload Audio & Generate Lipsync
        # ============================================================
        print("\n" + "="*60)
        print("STEP 3/3: Uploading Audio & Generating Lipsync...")
        print("="*60)
        
        # Upload audio to tmpfiles.org
        with langsmith.trace(name="step3a_upload_audio", tags=["audio-upload"]) as upload_trace:
            with open(audio_filename, 'rb') as f:
                files = {'file': f}
                upload_response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
                upload_response.raise_for_status()
                upload_data = upload_response.json()
                
                if upload_data.get('status') != 'success':
                    raise HTTPException(status_code=500, detail="Failed to upload audio")
                
                temp_url = upload_data['data']['url']
                audio_url = temp_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                upload_trace.outputs = {"audio_url": audio_url}
        
        print(f"✅ Audio Uploaded: {audio_url}")
        
        # Generate lipsynced video
        lipsynced_url = None
        
        with langsmith.trace(name="step3b_lipsync", tags=["lipsync"]) as lipsync_trace:
            lipsync_agent = create_lipsync_agent()
            
            task4 = Task(
                description=f"""Use Lipsync Video Generator tool.

Parameters:
  video_url: {video_url}
  audio_url: {audio_url}
  output_filename: lipsynced_{conversation_id}""",
                expected_output="Lipsynced video URL",
                agent=lipsync_agent,
                human_input=False
            )
            
            crew4 = Crew(agents=[lipsync_agent], tasks=[task4], verbose=True, process="sequential")
            lipsync_result = crew4.kickoff()
            lipsync_result_str = str(lipsync_result)
            lipsync_trace.outputs = {"result": lipsync_result_str[:500]}
            
            # Extract lipsynced URL
            if "Success! Output URL:" in lipsync_result_str:
                parts = lipsync_result_str.split("Success! Output URL:")
                if len(parts) > 1:
                    url_part = parts[1].strip()
                    lipsynced_url = url_part.split()[0] if url_part else None
        
        if not lipsynced_url:
            raise HTTPException(status_code=500, detail="Lipsynced video URL not generated")
        
        print(f"✅ Lipsynced Video Generated: {lipsynced_url}")
        
        execution_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Complete Workflow Finished in {execution_time:.2f} seconds")
        print(f"{'='*60}")
        print(f"Script: {script_filename}")
        print(f"Audio: {audio_filename}")
        print(f"Video: {video_url}")
        print(f"Lipsynced: {lipsynced_url}")
        print(f"{'='*60}\n")
        
        # Add to conversation
        conversations[conversation_id].append({
            "role": "assistant",
            "message": f"Generated complete lipsynced video for {request.ugc_image_path}",
            "script": video_script,
            "dialogue": dialogue,
            "audio_file": audio_filename,
            "video_url": video_url,
            "lipsynced_video_url": lipsynced_url,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get trace URL
        if run_tree and run_tree.id:
            try:
                tenant_id = langsmith_client._get_tenant_id()
                project_name = os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator")
                trace_url = f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project_name}/r/{run_tree.id}"
            except Exception as e:
                print(f"Could not generate trace URL: {e}")
        
        return LipsyncVideoResponse(
            conversation_id=conversation_id,
            script=video_script,
            dialogue=dialogue,
            audio_file=audio_filename,
            video_url=video_url,
            lipsynced_video_url=lipsynced_url,
            ugc_image_path=request.ugc_image_path,
            avatar_id=request.avatar_id,
            voice_used=voice_name,
            timestamp=datetime.now().isoformat(),
            trace_url=trace_url
        )
        
    except Exception as e:
        print(f"Error in complete video generation: {str(e)}")
        if run_tree and run_tree.id:
            langsmith_client.create_feedback(run_tree.id, key="error", score=0.0, comment=f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/feedback/{conversation_id}")
async def submit_feedback(
    conversation_id: str,
    feedback_type: str,  # "thumbs_up" or "thumbs_down"
    run_id: Optional[str] = None,
    comment: Optional[str] = None
):
    """
    Submit user feedback for a conversation
    """
    try:
        if not run_id:
            return {"status": "error", "message": "run_id required"}

        score = 1.0 if feedback_type == "thumbs_up" else 0.0

        langsmith_client.create_feedback(
            run_id,
            key=feedback_type,
            score=score,
            comment=comment
        )

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "feedback_type": feedback_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Starting UGC Orchestrator API Server")
    print("="*60)
    print("📡 Endpoint: http://localhost:8000/chat/ugc")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔍 LangSmith Tracing:", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    print("📊 Project:", os.getenv("LANGCHAIN_PROJECT", "ugc-orchestrator"))
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


