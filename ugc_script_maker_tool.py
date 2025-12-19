"""
UGC Script Maker Tool for CrewAI
Generates Veo-3-compatible 8-second video scripts using GPT-5.2
"""

import os
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class UGCScriptMakerInput(BaseModel):
    """Input schema for UGC Script Maker Tool."""
    ugc_image_reference: str = Field(
        ...,
        description="Path or URL to the UGC image reference"
    )
    product_name: str = Field(
        ...,
        description="Name of the product to feature in the video"
    )
    tone: str = Field(
        ...,
        description="Desired tone for the video (e.g., energetic, calm, professional, playful)"
    )
    platform: str = Field(
        ...,
        description="Target platform for the video (e.g., TikTok, Instagram, YouTube Shorts)"
    )


class UGCScriptMakerTool(BaseTool):
    name: str = "UGC Script Maker"
    description: str = (
        "Generates a highly detailed, Veo-3-compatible 8-second video script "
        "based on a selected UGC image."
    )
    args_schema: Type[BaseModel] = UGCScriptMakerInput
    cache_function: bool = False
    
    def _run(
        self,
        ugc_image_reference: str,
        product_name: str,
        tone: str,
        platform: str
    ) -> str:
        """
        Generate an 8-second video script optimized for Veo 3.
        
        Args:
            ugc_image_reference: Path or URL to the UGC image
            product_name: Name of the product
            tone: Desired tone for the video
            platform: Target platform
            
        Returns:
            Structured 8-second video script
        """
        # Initialize OpenAI client with AI/ML API
        api_key = os.getenv("AIML_API_KEY")
        if not api_key:
            return "Error: AIML_API_KEY environment variable not set"
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com/v1"
        )
        
        # System prompt for GPT-5.2
        system_prompt = '''You are a professional UGC commercial video director and script supervisor.

Your task:
- Generate a SINGLE, production-ready video script optimized for Veo 3
- The script must feel like authentic creator-made UGC, not a polished advertisement
- The script must strictly follow the output format below
- Output ONLY the script, no explanations, no markdown

GLOBAL RULES (CRITICAL):
- Assume the person, face, clothing, hair, and environment come directly from the provided UGC image reference
- Do NOT invent new identities, outfits, locations, props, or backgrounds
- The video is ALWAYS silent (no speech audio, no music, no ambient sound, no captions)
- Lip motion may be present but is visual-only
- The product must NEVER approach the mouth or face
- No sipping, no pretending to drink
- Everything described must be physically filmable

PLATFORM & FORMAT RULES:
- Default format: Vertical 9:16 (Instagram Stories / Reels)
- Respect Instagram safe areas (avoid extreme top and bottom edges)
- Duration must match the requested duration exactly

LIP MOTION CONSTRAINT (MANDATORY):
- Subtle, continuous speech-mimicking lip motion must persist from 0 seconds through the final frame
- Lip motion must NOT stop, pause, or freeze at any point, including the ending frame
- Lip motion continues even if body, camera, or product motion settles

PRODUCT VISIBILITY & TEXT SAFETY (MANDATORY):
- The product must NEVER be fully visible
- Only a partial section of the product may appear in frame at any time
- Fingers, framing, angle, and/or crop must naturally obscure fine printed text, nutrition labels, and barcodes
- Only brand colors and a partial logo may be visible
- Never request readable fine print

PRODUCT HANDLING RULES:
- Product stays at chest or torso level
- Grip must look relaxed and natural
- Product handling must feel casual, not deliberately showcased

CAMERA & MOTION STYLE:
- Medium close-up unless specified otherwise
- Subtle handheld realism
- Gentle push-in or static framing only
- No fast pans, no aggressive movement

CLIP CHAINING RULES (IMPORTANT):
- The ending frame may become more stable to allow seamless continuation into the next clip
- Product motion may settle near the end
- Lip motion must still continue through the final frame

EXPRESSION & PERFORMANCE:
- Energetic but natural creator demeanor
- No exaggerated facial expressions
- No wide smiles, laughter, or comedic acting
- Eye contact with the camera should feel confident and intentional

OUTPUT FORMAT (STRICT — FOLLOW EXACTLY):

FORMAT:
DURATION:
OUTPUT CONSTRAINT:
SCENE:
CAMERA:
SUBJECT:
ACTION TIMELINE:
  0–2s:
  2–4s:
  4–6s:
  6–8s:
PRODUCT HANDLING CONSTRAINT:
PRODUCT VISIBILITY:
DEPTH OF FIELD:
SUBTLE MOTION:
ENVIRONMENT:
LIGHTING:
MOTION DETAILS:
ENDING FRAME:

'''
        
        # User prompt with context
        user_prompt = f"""Generate an 8-second video script with the following parameters:

UGC Image Reference: {ugc_image_reference}
Product Name: {product_name}
Tone: {tone}
Platform: {platform}

Use the visual identity from the UGC image reference. Create a script that showcases the product naturally and authentically."""
        
        try:
            # Call GPT-5.2 via AI/ML API
            response = client.chat.completions.create(
                model="openai/gpt-5-2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
            )
            
            script = response.choices[0].message.content
            return script
            
        except Exception as e:
            return f"Error generating script: {str(e)}"


# Example usage
if __name__ == "__main__":
    tool = UGCScriptMakerTool()
    
    result = tool._run(
        ugc_image_reference="ugc_fb54f3f9-aefa-474d-b6ac-9f82a41145a4_20251214_121128.png",
        product_name="starbucks coffe",
        tone="calm",
        platform="instagram stories"
    )
    
    print(result)
