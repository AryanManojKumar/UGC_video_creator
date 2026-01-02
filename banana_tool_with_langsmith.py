import requests
import base64
import os
from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
import langsmith
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class BananaUGCInput(BaseModel):
    """Input schema for BananaUGCTool."""
    person_image_path: str = Field(..., description="Path to the person image file")
    product_image_path: str = Field(..., description="Path to the product image file")
    prompts: List[str] = Field(
        ...,
        description="List of 4 prompt variants for parallel image generation"
    )

class BananaUGCTool(BaseTool):
    name: str = "Banana UGC Image Generator"
    description: str = "Generates 4 UGC-style images in parallel where a person is showcasing a product. Takes a list of 4 prompts and generates all images concurrently."
    args_schema: Type[BaseModel] = BananaUGCInput

    @traceable(
        name="banana_ugc_tool_parallel",
        tags=["image-generation", "banana-api", "ugc", "parallel"],
        metadata={"model": "google/nano-banana-pro-edit", "provider": "aiml-api", "num_images": 4}
    )
    def _run(
        self,
        person_image_path: str,
        product_image_path: str,
        prompts: List[str]
    ) -> str:
        """
        Generate 4 UGC images in parallel using AI/ML API with google/nano-banana-pro model.
        Each prompt gets its own API call, all executed concurrently.
        """
        if len(prompts) != 4:
            return f"Error: Expected exactly 4 prompts, got {len(prompts)}"

        api_key = os.getenv("AIML_API_KEY")
        if not api_key:
            return "Error: AIML_API_KEY not found in environment variables"

        # Read and encode images once (shared across all calls)
        with langsmith.trace(
            name="encode_input_images",
            inputs={
                "person_image_path": person_image_path,
                "product_image_path": product_image_path
            },
            tags=["image-processing", "base64-encoding"],
            metadata={}
        ) as encode_trace:
            try:
                with open(person_image_path, "rb") as f:
                    person_image_data = f.read()
                    person_image_b64 = base64.b64encode(person_image_data).decode()

                with open(product_image_path, "rb") as f:
                    product_image_data = f.read()
                    product_image_b64 = base64.b64encode(product_image_data).decode()

                encode_trace.metadata.update({
                    "person_image_size": len(person_image_data),
                    "product_image_size": len(product_image_data),
                    "person_image_b64_length": len(person_image_b64),
                    "product_image_b64_length": len(product_image_b64)
                })
                encode_trace.outputs = {"status": "images_encoded"}

            except FileNotFoundError as e:
                error_msg = f"Error: Image file not found - {str(e)}"
                encode_trace.outputs = {"error": error_msg}
                return error_msg

        # Generate images in parallel
        print(f"\n🚀 Starting parallel generation of 4 images...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all 4 API calls concurrently
            futures = {}
            for idx, prompt in enumerate(prompts, 1):
                future = executor.submit(
                    self._generate_single_image,
                    person_image_b64,
                    product_image_b64,
                    prompt,
                    f"generated_ugc_image_{idx}.png",
                    idx,
                    api_key
                )
                futures[future] = idx

            # Collect results as they complete
            results = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    print(f"✅ Image {idx}/4 completed")
                except Exception as e:
                    error_msg = f"Error generating image {idx}: {str(e)}"
                    results[idx] = {"success": False, "error": error_msg}
                    print(f"❌ Image {idx}/4 failed: {error_msg}")

        total_time = time.time() - start_time
        print(f"\n⚡ All 4 images generated in {total_time:.2f}s (parallel)")

        # Format response
        success_count = sum(1 for r in results.values() if r.get("success"))
        
        if success_count == 4:
            response = "✅ SUCCESS: All 4 images generated in parallel!\n\n"
            for idx in sorted(results.keys()):
                response += f"Image {idx}: {results[idx]['output_path']}\n"
            response += f"\nTotal time: {total_time:.2f}s"
            return response
        else:
            response = f"⚠️ PARTIAL SUCCESS: {success_count}/4 images generated\n\n"
            for idx in sorted(results.keys()):
                if results[idx].get("success"):
                    response += f"✅ Image {idx}: {results[idx]['output_path']}\n"
                else:
                    response += f"❌ Image {idx}: {results[idx].get('error', 'Unknown error')}\n"
            return response

    @traceable(
        name="generate_single_image",
        tags=["image-generation", "banana-api", "single-call"],
        metadata={"model": "google/nano-banana-pro-edit"}
    )
    def _generate_single_image(
        self,
        person_image_b64: str,
        product_image_b64: str,
        prompt: str,
        output_filename: str,
        image_index: int,
        api_key: str
    ) -> dict:
        """
        Generate a single UGC image (called in parallel by ThreadPoolExecutor).
        Returns dict with success status and output path or error.
        """

        # AI/ML API endpoint
        url = "https://api.aimlapi.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Construct prompt for image-to-image composition
        full_prompt = (
            f"Combine these images to create a realistic UGC-style photo where the person "
            f"from the first image is naturally showcasing the product from the second image. "
            f"{prompt}. Keep the same person's face and identity, and the exact product appearance. "
            f"Make it look like authentic user-generated content."
        )

        # Use correct schema for google/nano-banana-pro-edit
        payload = {
            "model": "google/nano-banana-pro-edit",
            "prompt": full_prompt,
            "image_urls": [
                f"data:image/jpeg;base64,{person_image_b64}",
                f"data:image/jpeg;base64,{product_image_b64}"
            ],
            "aspect_ratio": "1:1",
            "resolution": "1K",
            "num_images": 1
        }

        # Call image generation API with detailed tracing
        with langsmith.trace(
            name=f"call_banana_api_image_{image_index}",
            inputs={
                "model": "google/nano-banana-pro-edit",
                "prompt": full_prompt,
                "image_index": image_index
            },
            tags=["api-call", "image-generation", "aiml-api"],
            metadata={
                "provider": "AI/ML API",
                "endpoint": url,
                "model": "google/nano-banana-pro-edit",
                "image_index": image_index
            }
        ) as api_trace:
            try:
                call_start = time.time()

                print(f"   🎨 Calling API for image {image_index}...")
                
                response = requests.post(url, json=payload, headers=headers, timeout=180)

                latency = time.time() - call_start

                # Log API call metadata
                api_trace.metadata.update({
                    "status_code": response.status_code,
                    "latency_seconds": round(latency, 2),
                    "response_size_bytes": len(response.content)
                })

                # Accept both 200 and 201 status codes
                if response.status_code not in [200, 201]:
                    error_msg = f"API returned status {response.status_code}: {response.text[:500]}"
                    api_trace.outputs = {"error": error_msg, "status_code": response.status_code}
                    return {"success": False, "error": error_msg}

                result = response.json()
                api_trace.outputs = {"status": "success", "result_keys": list(result.keys())}

                # Estimate cost
                estimated_cost = 0.02
                api_trace.metadata["estimated_cost_usd"] = estimated_cost

            except requests.exceptions.Timeout:
                error_msg = f"API request timed out after 180 seconds"
                api_trace.outputs = {"error": error_msg}
                return {"success": False, "error": error_msg}
            except requests.exceptions.RequestException as e:
                error_msg = f"Error calling AI/ML API: {str(e)[:200]}"
                api_trace.outputs = {"error": error_msg}
                return {"success": False, "error": error_msg}

        # Save the generated image
        with langsmith.trace(
            name=f"save_generated_image_{image_index}",
            tags=["image-output", "file-save"],
            metadata={"image_index": image_index}
        ) as save_trace:
            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]
                output_path = output_filename

                # Check if it's a URL or base64
                if "url" in image_data:
                    # Download from URL
                    img_response = requests.get(image_data["url"])
                    img_response.raise_for_status()

                    with open(output_path, "wb") as f:
                        f.write(img_response.content)

                    save_trace.metadata.update({
                        "method": "url_download",
                        "image_url": image_data["url"],
                        "output_path": output_path,
                        "image_size_bytes": len(img_response.content)
                    })
                    save_trace.outputs = {"output_path": output_path, "url": image_data["url"]}

                    return {"success": True, "output_path": output_path}

                elif "b64_json" in image_data:
                    # Decode base64
                    image_bytes = base64.b64decode(image_data["b64_json"])

                    with open(output_path, "wb") as f:
                        f.write(image_bytes)

                    save_trace.metadata.update({
                        "method": "base64_decode",
                        "output_path": output_path,
                        "image_size_bytes": len(image_bytes)
                    })
                    save_trace.outputs = {"output_path": output_path}

                    return {"success": True, "output_path": output_path}

                else:
                    error_msg = f"Unexpected image format in response"
                    save_trace.outputs = {"error": error_msg}
                    return {"success": False, "error": error_msg}
            else:
                error_msg = f"No image data in response"
                save_trace.outputs = {"error": error_msg}
                return {"success": False, "error": error_msg}
