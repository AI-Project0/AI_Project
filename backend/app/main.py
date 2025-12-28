from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Literal
import uvicorn
import io
import asyncio

from app.core.ai_processor import AIProcessor

app = FastAPI(title="Pro-ID Gen API", version="1.0.0")

# CORS Setup
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Processor
ai_processor = AIProcessor()

@app.on_event("startup")
async def startup_event():
    # Helper to preload models if needed, though they load lazily in current impl
    pass

@app.get("/health")
def health_check():
    """ Health check endpoint for deployment (e.g. Render) """
    return {"status": "awake", "message": "Backend is ready"}

@app.post("/generate")
async def generate_id_photo(
    file: UploadFile = File(...),
    size_id: Literal["1inch", "2inch_head", "2inch_half"] = Form("2inch_head"),
    bg_color: str = Form("#FFFFFF"),
    outfit_type: Literal["original", "suit_male", "suit_female"] = Form("suit_male")
):
    """
    Generate an ID photo from an uploaded selfie.
    
    - **file**: The input image (selfie).
    - **size_id**: "1inch" (28:35), "2inch_head" (35:45), "2inch_half" (42:47).
    - **bg_color**: Hex string (e.g. #FFFFFF, #4B89DC).
    - **outfit_type**: "original" (no inpaint), "suit_male", "suit_female".
    """
    
    # Valdiate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    
    # --- Logic Mapping ---
    
    # 1. Size Logic (Crop Ratios)
    # Passed to AIProcessor as (width_ratio, height_ratio) or specific identifier
    # "1inch" -> 28:35
    # "2inch_head" -> 35:45 (Standard)
    # "2inch_half" -> 42:47 (?) - standard "2 inch half body" for some visa/resume?
    # Let's map directly to target aspect ratio float w/h
    
    size_map = {
        "1inch": (28, 35),
        "2inch_head": (35, 45),
        "2inch_half": (42, 47)
    }
    target_ratio = size_map.get(size_id, (35, 45))

    # 2. Outfit Logic (Prompt Construction)
    prompt = ""
    negative_prompt = ""
    
    # If original, we skip inpainting steps in the processor using a flag
    is_original_outfit = (outfit_type == "original")
    
    if outfit_type == "suit_male":
        prompt = "man wearing dark blue formal suit, white shirt, tie"
        negative_prompt = "open collar, casual, t-shirt"
    elif outfit_type == "suit_female":
        prompt = "woman wearing formal business blazer, white shirt, elegant"
        negative_prompt = "low cut, casual, pattern"
        
    try:
        # Call AI Processor
        # We need to update AIProcessor.generate_id_photo to accept these new params
        output_bytes = await ai_processor.generate_id_photo(
            image_bytes=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio_tuple=target_ratio,
            bg_color=bg_color,
            preserve_outfit=is_original_outfit
        )
        
        # Return as image
        return Response(content=output_bytes, media_type="image/png")
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
