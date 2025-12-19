"""
Veo-3.1 Image-to-Video Generator Tool for CrewAI
Converts: (image + script) -> Veo-3.1 Image-to-Video output
"""

import os
import base64
import time
import requests
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class Veo3VideoMakerInput(BaseModel):
    """Input schema for Veo3 Video Generator."""
    image_reference: str = Field(
        ...,
        description="URL or local path to reference image for Veo-3 video conditioning"
    )
    script_text: str = Field(
        ...,
        description="The final structured script to turn into a video"
    )
    duration_seconds: int = Field(
        default=8,
        description="Desired video duration (default 8 seconds)"
    )


class Veo3VideoMakerTool(BaseTool):
    name: str = "Veo3.1 Image-to-Video Generator"
    description: str = (
        "Converts a script + reference image into a generated video using "
        "Google Veo-3.1 Image-to-Video API."
    )
    args_schema: Type[BaseModel] = Veo3VideoMakerInput
    cache_function: bool = False

    def _get_image_url(self, path_or_url: str) -> str:
        """Converts local image to base64 data URI or returns URL directly."""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        
        # Read local file and convert to base64 data URI
        try:
            with open(path_or_url, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine MIME type from extension
            ext = path_or_url.lower().split('.')[-1]
            mime_map = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }
            mime_type = mime_map.get(ext, 'image/png')
            
            return f"data:{mime_type};base64,{image_data}"
        except Exception as e:
            raise Exception(f"Error reading local image file: {str(e)}")

    def _run(self, image_reference: str, script_text: str, duration_seconds: int) -> str:
        """Generate Veo-3.1 video from image + script."""

        api_key = os.getenv("AIML_API_KEY")
        if not api_key:
            return "Error: AIML_API_KEY environment variable not set"

        base_url = "https://api.aimlapi.com/v2"
        
        # Get image URL (converts local files to base64 data URI)
        try:
            image_url = self._get_image_url(image_reference)
        except Exception as e:
            return f"Error processing image: {str(e)}"

        # Step 1: Create video generation task
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "google/veo-3.1-i2v",
            "prompt": script_text,
            "image_url": image_url,
            "duration": duration_seconds,
            "aspect_ratio": "9:16",
            "resolution": "1080p",
            "generate_audio": False
        }

        try:
            # Submit generation request
            response = requests.post(
                f"{base_url}/video/generations",
                json=payload,
                headers=headers
            )
            
            if response.status_code >= 400:
                return f"Error submitting video generation: {response.status_code} - {response.text}"
            
            response_data = response.json()
            generation_id = response_data.get("id")
            
            if not generation_id:
                return f"Error: No generation ID returned. Response: {response_data}"
            
            print(f"Video generation started. ID: {generation_id}")
            
            # Step 2: Poll for completion
            timeout = 300  # 5 minutes timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                check_response = requests.get(
                    f"{base_url}/video/generations",
                    params={"generation_id": generation_id},
                    headers=headers
                )
                
                if check_response.status_code >= 400:
                    return f"Error checking status: {check_response.status_code} - {check_response.text}"
                
                result = check_response.json()
                status = result.get("status")
                
                print(f"Status: {status}")
                
                if status == "completed":
                    video_url = result.get("video", {}).get("url")
                    if video_url:
                        return f"✅ Video generated successfully: {video_url}\n\n[VIDEO_GENERATION_COMPLETE] - Task finished."
                    return f"Video completed but no URL found. Response: {result}"
                
                elif status == "failed":
                    error = result.get("error", "Unknown error")
                    return f"Video generation failed: {error}"
                
                elif status in ["waiting", "active", "queued", "generating"]:
                    time.sleep(10)
                else:
                    return f"Unknown status: {status}. Response: {result}"
            
            return f"Timeout: Video generation took longer than {timeout} seconds. Generation ID: {generation_id}"

        except Exception as e:
            return f"Error generating Veo-3.1 video: {str(e)}"


# Example direct run
if __name__ == "__main__":
    tool = Veo3VideoMakerTool()
    
    result = tool._run(
        image_reference="ugc_fb54f3f9-aefa-474d-b6ac-9f82a41145a4_20251214_121128.png",
        script_text='''FORMAT:
Vertical 9:16 (Instagram Stories)

DURATION:
8 seconds

OUTPUT CONSTRAINT:
Silent video (no speech audio, no music, no captions). Subtle, continuous speech-mimicking lip motion from 0s through the final frame without stopping. Product never approaches mouth/face. Only partial product visible at all times; fine print/barcodes/nutrition text must be naturally obscured.

SCENE:
Use the exact person, outfit, hair, and environment from the provided UGC image reference. Medium close-up framing with torso visible.

CAMERA:
Handheld smartphone feel, medium close-up, gentle push-in (very subtle) with stable horizon. Keep subject centered within Instagram safe areas.

SUBJECT:
The same creator from the reference image, calm demeanor, intentional eye contact with camera, relaxed posture.

ACTION TIMELINE:
  0–2s:
Subject faces camera with calm, steady eye contact and subtle continuous lip motion as if speaking softly. One hand holds the product at mid-chest level, but it’s mostly out of frame—only a corner/edge of the Starbucks coffee packaging peeks in from the lower-right side. Fingers naturally cover any text; only brand color and a partial logo shape may be hinted.
  2–4s:
A small, natural wrist adjustment brings a slightly different partial angle of the product into view (still only a portion visible). Subject gives a gentle, approving micro-nod while maintaining continuous lip motion. The other hand briefly smooths or taps the front of their shirt/torso area casually, then returns to a relaxed position.
  4–6s:
Subject glances down briefly toward the product (eyes dip for a moment), then returns to direct eye contact. Product remains at chest level; grip stays relaxed. The camera continues a barely noticeable push-in, keeping the product partially cropped and text obscured.
  6–8s:
Subject holds steady, calm expression, continuing subtle lip motion through the end. Product motion settles with only a small natural sway from handheld movement. Maintain partial product visibility at the lower edge/side of frame; no deliberate “showing” gesture.

PRODUCT HANDLING CONSTRAINT:
Product must remain at chest/torso level the entire time. No lifting toward face. No sipping or drinking. Grip relaxed; no pointing at labels.

PRODUCT VISIBILITY:
Never fully visible. Only a partial section appears in frame at any moment. Fingers and cropping must obscure fine printed text, nutrition labels, and barcodes; only brand colors and a partial logo may be visible. 

DEPTH OF FIELD:
Natural smartphone depth; subject in clear focus, background slightly soft but recognizable as the same environment from the reference.

SUBTLE MOTION:
Continuous speech-mimicking lip motion throughout. Minimal head movement (micro-nods). Gentle handheld sway; very subtle push-in.

ENVIRONMENT:
Exactly as in the reference image—no new props or background changes introduced.

LIGHTING:
Match the reference image lighting (same direction, softness, and color temperature). No dramatic changes. 

MOTION DETAILS:
Keep movements calm and unhurried. Avoid fast pans. Maintain stable framing within safe areas; product stays partially cropped and naturally obscured.

ENDING FRAME:
Camera becomes slightly more stable for the last half-second; subject maintains direct eye contact and continuous subtle lip motion through the final frame. Product remains partially visible at chest level with text still obscured, ready for seamless continuation.''',  # your script output here
        duration_seconds=8
    )

    print(result)
