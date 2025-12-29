from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Literal
import uvicorn
import io
import asyncio

from app.core.ai_processor import AIProcessor

app = FastAPI(title="Pro-ID Gen API", version="1.0.0")

# 修改 main.py 的 origins 部分
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # <--- 暫時加入這個，允許所有網址連線 (方便測試)
]
# 或者更安全的寫法是之後把 Render 的前端網址加進來

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Processor
ai_processor = AIProcessor()

# Standard ID photo sizes (width_mm, height_mm)
size_map = {
    "1inch": (28, 35),
    "2inch_head": (35, 45),
    "2inch_half": (42, 47)
}

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
    
    # --- DEBUG LOGGING ---
    print(f"--- Incoming Request to /generate ---")
    print(f"File: {file.filename}, Content-Type: {file.content_type}")
    print(f"Size ID: {size_id}")
    print(f"BG Color: {bg_color}")
    print(f"Outfit Type: {outfit_type}")
    print(f"------------------------------------")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        msg = f"File must be an image. Received: {file.content_type}"
        print(f"400 Error: {msg}")
        raise HTTPException(status_code=400, detail=msg)
    
    image_bytes = await file.read()
    
    # 1. Size Logic (Crop Ratios)
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
        print(f"400 ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/analyze-face")
async def analyze_face(
    file: UploadFile = File(...),
    size_id: Literal["1inch", "2inch_head", "2inch_half"] = Form("2inch_head")
):
    """
    Analyze the uploaded image and return a recommended crop box.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    
    target_ratio_tuple = size_map.get(size_id, (35, 45))
    target_ratio_float = target_ratio_tuple[0] / target_ratio_tuple[1]
    
    result = ai_processor.analyze_face(image_bytes, target_ratio_float)
    
    if not result:
        raise HTTPException(status_code=400, detail="No face detected or analysis failed.")
        
    return result

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
