# UGC Person-to-Video Pipeline

An AI-powered pipeline that transforms person images into authentic User-Generated Content (UGC) videos. Upload a person photo and product image, and the system generates 4 diverse UGC images, creates video scripts, and produces professional videos using Google's Veo-3.1 model.

## What It Does

This pipeline automates the entire UGC video creation workflow:

1. **Image Generation**: Combines person + product photos into 4 diverse UGC-style images
2. **Script Creation**: Generates professional 8-second video scripts optimized for social media
3. **Video Production**: Converts images + scripts into videos using Google Veo-3.1

## Features

- **Multi-Agent Architecture**: Specialized AI agents handle prompts, images, scripts, and videos
- **4 Image Variants**: Automatically generates diverse poses, angles, and compositions
- **Platform-Optimized**: Scripts tailored for TikTok, Instagram, YouTube Shorts
- **LangSmith Monitoring**: Full observability and tracing for all AI operations
- **REST API**: FastAPI server with chat-based interface
- **Web Interface**: Simple HTML frontend for easy interaction

## Tech Stack

- **AI Models**: GPT-5 (via AI/ML API), Google Nano Banana Pro Edit, Google Veo-3.1
- **Framework**: CrewAI for multi-agent orchestration
- **API**: FastAPI with CORS support
- **Monitoring**: LangSmith for tracing and observability
- **Language**: Python 3.8+

## Architecture

### Agent Workflow

```
User Input (Person + Product Images)
    ↓
Prompt Agent → Generates 4 diverse prompts
    ↓
Image Agent → Creates 4 UGC images (one per prompt)
    ↓
Script Agent → Generates 8-second video script
    ↓
Video Agent → Produces final video with Veo-3.1
```

### Core Components

- **ugc_orchestrator_agent.py**: Main orchestrator coordinating all agents
- **prompt_agent.py**: Generates diverse prompt variations
- **image_generator_agent.py**: Creates UGC images using Banana API
- **script_agent.py**: Writes video scripts optimized for Veo-3
- **video_agent.py**: Generates videos using Google Veo-3.1
- **server_with_langsmith.py**: FastAPI server with full tracing

## Installation

### Prerequisites

- Python 3.8 or higher
- AI/ML API key (get from [aimlapi.com](https://aimlapi.com))
- LangSmith API key (optional, for monitoring)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ugc-pipeline
```

2. Install dependencies:
```bash
pip install crewai fastapi uvicorn python-dotenv openai requests pydantic langsmith
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
AIML_API_KEY=your_aiml_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=ugc-orchestrator
```

## Usage

### Start the Server

```bash
python server_with_langsmith.py
```

The server starts at `http://localhost:8000`

### Web Interface

Open your browser to `http://localhost:8000` to access the chat interface.

### API Endpoints

#### Generate UGC Images

```bash
POST /chat/ugc/upload
```

Upload person and product images to generate 4 UGC variants:

```bash
curl -X POST http://localhost:8000/chat/ugc/upload \
  -F "message=Create authentic UGC content" \
  -F "person_image=@person.jpg" \
  -F "product_image=@product.jpg"
```

Response:
```json
{
  "conversation_id": "uuid",
  "assistant_message": "Successfully generated 4 images...",
  "generated_images": [
    "ugc_uuid_timestamp_1.png",
    "ugc_uuid_timestamp_2.png",
    "ugc_uuid_timestamp_3.png",
    "ugc_uuid_timestamp_4.png"
  ],
  "trace_url": "https://smith.langchain.com/..."
}
```

#### Generate Script and Video

```bash
POST /chat/ugc/script
```

Create a video from a generated UGC image:

```bash
curl -X POST http://localhost:8000/chat/ugc/script \
  -H "Content-Type: application/json" \
  -d '{
    "ugc_image_path": "ugc_uuid_timestamp_1.png",
    "product_name": "Starbucks Coffee",
    "tone": "energetic",
    "platform": "Instagram"
  }'
```

Response:
```json
{
  "conversation_id": "uuid",
  "script": "FORMAT:\nVertical 9:16...",
  "video_url": "https://...",
  "ugc_image_path": "ugc_uuid_timestamp_1.png",
  "trace_url": "https://smith.langchain.com/..."
}
```

### Direct Python Usage

```python
from ugc_orchestrator_agent import generate_ugc_with_orchestrator

# Generate 4 UGC images
result = generate_ugc_with_orchestrator(
    person_image_path="person.jpg",
    product_image_path="product.jpg",
    base_intent="A happy person showcasing the product naturally"
)

print(result)
```

## API Documentation

Full interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### Model Settings

Edit the agent files to customize AI models:

```python
# In any agent file
llm = LLM(
    model="openai/gpt-5-2025-08-07",
    api_key=os.getenv("AIML_API_KEY"),
    base_url="https://api.aimlapi.com/v1",
    temperature=0.7  # Adjust creativity
)
```

### Video Settings

Modify video parameters in `ugc_vid_maker_tool.py`:

```python
payload = {
    "model": "google/veo-3.1-i2v",
    "duration": 8,  # seconds
    "aspect_ratio": "9:16",  # vertical
    "resolution": "1080p",
    "generate_audio": False
}
```

## Monitoring with LangSmith

All agent operations are traced in LangSmith for debugging and optimization:

1. View traces at `https://smith.langchain.com`
2. Each API response includes a `trace_url` for direct access
3. Monitor token usage, latency, and costs
4. Debug agent reasoning and tool calls

## Project Structure

```
.
├── ugc_orchestrator_agent.py    # Main orchestrator
├── prompt_agent.py               # Prompt variation agent
├── image_generator_agent.py     # Image generation agent
├── script_agent.py               # Script writing agent
├── video_agent.py                # Video generation agent
├── banana_tool_with_langsmith.py # Image generation tool
├── prompt_variator_tool.py      # Prompt variation tool
├── ugc_script_maker_tool.py     # Script generation tool
├── ugc_vid_maker_tool.py        # Video generation tool
├── server_with_langsmith.py     # FastAPI server
├── chatbox.html                  # Web interface
├── .env.example                  # Environment template
└── README.md                     # This file
```

## Workflow Details

### Image Generation Workflow

1. **Prompt Variator** generates 4 diverse prompts from base intent
2. **Image Generator** creates 4 images using:
   - Model: `google/nano-banana-pro-edit`
   - Input: Person image + Product image + Prompt
   - Output: 4 PNG files with unique compositions

### Video Generation Workflow

1. **Script Generator** creates 8-second script using GPT-5.2:
   - Analyzes UGC image reference
   - Optimizes for target platform
   - Includes detailed camera, motion, and timing instructions

2. **Video Generator** produces video using Veo-3.1:
   - Input: UGC image + Script
   - Duration: 8 seconds
   - Format: 9:16 vertical (1080p)
   - Output: Video URL

## Troubleshooting

### Common Issues

**API Key Errors**
```
Error: AIML_API_KEY not found
```
Solution: Ensure `.env` file exists with valid `AIML_API_KEY`

**Image Not Found**
```
Error: Image file not found
```
Solution: Verify image paths are correct and files exist

**Video Generation Timeout**
```
Timeout: Video generation took longer than 300 seconds
```
Solution: Video generation can take 3-5 minutes. Check generation ID in logs.

### Debug Mode

Enable verbose logging:
```python
# In agent files
agent = Agent(
    ...
    verbose=True  # Shows detailed agent reasoning
)
```

## Performance

- **Image Generation**: ~30-60 seconds per image (4 images in parallel)
- **Script Generation**: ~5-10 seconds
- **Video Generation**: ~3-5 minutes (Veo-3.1 processing)
- **Total Pipeline**: ~5-7 minutes for complete workflow

## Cost Estimates

Approximate costs per generation (varies by provider):
- Image generation: $0.02-0.05 per image
- Script generation: $0.01-0.02 per script
- Video generation: $0.10-0.20 per video

## Limitations

- Video generation requires significant processing time
- Image quality depends on input photo quality
- Best results with clear, well-lit person and product photos
- Videos are silent (no audio generation)

## Contributing

Contributions welcome! Areas for improvement:
- Add audio generation to videos
- Support batch processing
- Add more video duration options
- Implement caching for faster regeneration
- Add video editing capabilities

## License

[Add your license here]

## Support

For issues or questions:
- Check API documentation at [aimlapi.com/docs](https://aimlapi.com/docs)
- Review LangSmith traces for debugging
- Open an issue in the repository

## Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI)
- Powered by [AI/ML API](https://aimlapi.com)
- Monitored with [LangSmith](https://smith.langchain.com)
- Uses Google Veo-3.1 and Nano Banana Pro Edit models
