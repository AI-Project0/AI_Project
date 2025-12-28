import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import rembg
import mediapipe as mp
from typing import Optional, Tuple, List
import io

class AIProcessor:
    """
    Core AI Processor for Pro-ID Gen.
    Handles the entire pipeline from raw selfie to professional ID photo.
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIProcessor, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        print("Initializing AI Processor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        # Initialize Rembg
        self.rembg_session = rembg.new_session()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.initialized = True

    def load_models(self):
        """
        Loads the Stable Diffusion Inpainting model into memory.
        Using 'runwayml/stable-diffusion-inpainting' for reliability.
        """
        if self.pipe is not None:
            return

        print(f"Loading models to {self.device}...")
        
        try:
            # Load Stable Diffusion Inpainting Model
            # Using fp16 for GPU to save memory and increase speed
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch_dtype,
                safety_checker=None # Disable for speed and avoiding false positives on harmless ID photos
            )
            
            # Use DPMSolver for faster inference (20-30 steps)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            self.pipe.to(self.device)
            if self.device == "cuda":
                self.pipe.enable_attention_slicing() # Optimize memory
            
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e

    async def generate_id_photo(
        self, 
        image_bytes: bytes, 
        prompt: str, 
        negative_prompt: str = "",
        aspect_ratio_tuple: Tuple[int, int] = (35, 45),
        bg_color: str = "#FFFFFF",
        preserve_outfit: bool = False
    ) -> bytes:
        """
        Main pipeline execution.
        """
        if not preserve_outfit:
             self.load_models()
        
        # 0. Decode Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

        # Step A: Preprocessing (Rembg + Centering + Smart Crop)
        # Note: We now pass the target aspect ratio
        processed_img, face_landmarks = self._step_a_preprocessing(image, aspect_ratio_tuple)
        
        if not face_landmarks:
            raise ValueError("No face detected in the image.")

        if preserve_outfit:
            # Skip Generation steps (B & C)
            print("Preserving outfit. Skipping Inpainting.")
            # Ensure processed_img has transparency for compositing
            # The preprocessing currently returns white bg. We need it to return transparent for Steps B/C logic?
            # Actually _step_a_preprocessing currently composites to WHITE.
            # We should modify Step A to return RGBA or handle it better.
            
            # Let's modify Step A logic below to return RGBA (transparent) 
            # OR we just re-run rembg here? No, efficient to do it once.
            
            # Since I can't easily change Step A's return type for everyone without breaking logic,
            # Let's rely on Step A's Rembg behavior. 
            # BUT _step_a calculates crop on "clean_image" which is white BG.
            # We need the crop coordinates applied to the TRANSPARENT image too.
            
            # For simplicity in this iteration: Step A returns RGBA logic.
            # Let's trust Step A update below.
            generated_img = processed_img
        else:
            # Step B: Masking (Face Protection Mask)
            mask_img = self._step_b_masking(processed_img, face_landmarks)
            
            # Step C: Generation (SD Inpainting)
            generated_img = self._step_c_generation(processed_img, mask_img, prompt, negative_prompt)
        
        # Step D: Post-processing (Final Composition with specific BG color)
        final_img = self._step_d_postprocessing(generated_img, bg_color)
        
        # Return bytes
        output_buffer = io.BytesIO()
        final_img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def _step_a_preprocessing(self, image: Image.Image, aspect_ratio: Tuple[int, int] = (35, 45)) -> Tuple[Image.Image, List]:
        """
        Removes background and crops image to target ratio centered on face.
        Returns: RGBA Image (Transparent Background) - Critical for later compositing.
        """
        print(f"Running Step A: Preprocessing with ratio {aspect_ratio}...")
        
        # 1. Remove background
        no_bg_image = rembg.remove(image, session=self.rembg_session)
        # no_bg_image is RGBA
        
        # For landmarks, we need an RGB image (MediaPipe doesn't like Alpha sometimes or needs contrast)
        # Composite against white for detection
        background = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        background.paste(no_bg_image, mask=no_bg_image.split()[3])
        rgb_for_detection = background
        
        # 2. Detect Face Landmarks
        img_np = np.array(rgb_for_detection)
        results = self.face_mesh.process(img_np)
        
        if not results.multi_face_landmarks:
            return no_bg_image, [] # Return original if fail
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 3. Smart Crop
        w, h = no_bg_image.size # Use original size
        
        # Get face bounding box
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2
        face_height = max_y - min_y
        
        # Target Ratio calculations
        # aspect_ratio is (width_unit, height_unit), e.g., (35, 45) -> 0.777
        target_r = aspect_ratio[0] / aspect_ratio[1]
        
        # Crop Height logic: Face should be ~45-50% of height
        target_crop_h = face_height * 2.2 
        target_crop_w = target_crop_h * target_r
        
        # Center horizontally, Offset vertically
        left = face_center_x - (target_crop_w / 2)
        top = face_center_y - (target_crop_h * 0.45) 
        right = left + target_crop_w
        bottom = top + target_crop_h
        
        # Expand canvas to prevent cutting off if crop is out of bounds
        max_dim = int(max(w, h, target_crop_w, target_crop_h) * 2)
        
        # Create a large transparent canvas and paste the no_bg_image in center
        expanded_canvas = Image.new("RGBA", (max_dim, max_dim), (0,0,0,0))
        paste_x = (max_dim - w) // 2
        paste_y = (max_dim - h) // 2
        expanded_canvas.paste(no_bg_image, (paste_x, paste_y))
        
        # Adjust crop coordinates to new canvas
        crop_box = (
            int(left + paste_x),
            int(top + paste_y),
            int(right + paste_x),
            int(bottom + paste_y)
        )
        
        cropped_image = expanded_canvas.crop(crop_box)
        
        # Resize to standardized high-res base
        # e.g. Width 512, Height derived
        base_width = 512
        base_height = int(base_width / target_r)
        # Ensure divisible by 8 for SD
        base_height = (base_height // 8) * 8
        
        resized_image = cropped_image.resize((base_width, base_height), Image.Resampling.LANCZOS)
        
        # Re-detect landmarks on RESIZED image for next steps
        # Need RGB again for detection
        bg_white_resized = Image.new("RGB", resized_image.size, (255, 255, 255))
        if resized_image.mode == "RGBA":
            bg_white_resized.paste(resized_image, mask=resized_image.split()[3])
        else:
            bg_white_resized = resized_image.convert("RGB")
            
        img_np_new = np.array(bg_white_resized)
        results_new = self.face_mesh.process(img_np_new)
        
        new_landmarks = results_new.multi_face_landmarks[0].landmark if results_new.multi_face_landmarks else []
        
        return resized_image, new_landmarks

    def _step_b_masking(self, image: Image.Image, landmarks: List) -> Image.Image:
        """
        create a mask where:
        WHITE (255) = Area to INPAINT (Background, Body, Clothes)
        BLACK (0) = Area to KEEP (Face skin, inner features)
        """
        print("Running Step B: Masking...")
        w, h = image.size
        
        mask = Image.new("L", (w, h), 255) # Initialize as White
        draw = ImageDraw.Draw(mask)
        
        # Silhouette includes face and neck area
        silhouette_ids = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        if landmarks:
            points = []
            for idx in silhouette_ids:
                lm = landmarks[idx]
                points.append((lm.x * w, lm.y * h))
            draw.polygon(points, fill=0) # Protect Face
            
            # REFINED NECK PROTECTION
            # Use jawline landmarks (e.g., 172 on left, 397 on right) to define neck width
            # Index 152 is center chin.
            # We create a protection polygon that covers the neck more precisely down from the jaw.
            jaw_left = landmarks[172]
            jaw_right = landmarks[397]
            chin = landmarks[152]
            
            neck_points = [
                (jaw_left.x * w, jaw_left.y * h),
                (jaw_right.x * w, jaw_right.y * h),
                (jaw_right.x * w, (jaw_right.y + 0.08) * h), # Extend down
                (jaw_left.x * w, (jaw_left.y + 0.08) * h)
            ]
            draw.polygon(neck_points, fill=0)
            
            # Fill the chin to neck gap
            draw.ellipse([chin.x*w - 50, chin.y*h - 10, chin.x*w + 50, chin.y*h + 40], fill=0)

        # Apply Gaussian Blur to the mask to soften transitions (radius reduced for precision)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
        
        return mask

    def _step_c_generation(
        self, 
        image: Image.Image, 
        mask: Image.Image, 
        prompt: str, 
        negative_prompt: str
    ) -> Image.Image:
        """
        Run SD Inpainting
        """
        print(f"Running Step C: Generation... Prompt: {prompt}")
        
        # Convert RGBA to RGB (Neutral Grey Background) for SD Input
        # Neutral Grey (#808080) provides better contrast for shirt collars and reduces color bleed.
        if image.mode == "RGBA":
            base_image = Image.new("RGB", image.size, (128, 128, 128))
            base_image.paste(image, mask=image.split()[3])
        else:
            base_image = image.convert("RGB")

        # Enhance prompt for ID Photo
        # Positive Prompt ("Safe & Standard"): Simple and reliable professional ID photo
        base_prompt = "professional ID photo, asian person, wearing dark suit and white tie, clean white background, soft studio lighting, high quality, 4k"
        user_prompt_weighted = f"({prompt}:1.2)" 
        final_prompt = f"{base_prompt}, {user_prompt_weighted}"
        
        # Negative Prompt ("Safe & Standard"): Basic artifact and quality constraints
        base_negative = "(cartoon, anime, illustration:1.5), low quality, worst quality, blurry, deformed, distorted, messy, dirty face, shadows"
        final_negative = f"{base_negative}, {negative_prompt}"
        
        # Inference Tuning for "Safe & Standard" Result
        output = self.pipe(
            prompt=final_prompt,
            negative_prompt=final_negative,
            image=base_image,
            mask_image=mask,
            num_inference_steps=30,
            guidance_scale=6.5,
            strength=0.75
        ).images[0]
        
        return output

    def _step_d_postprocessing(self, image: Image.Image, bg_color_hex: str = "#FFFFFF") -> Image.Image:
        """
        Composite generated person onto solid Blue/White/Red background.
        If 'preserve_outfit' was used, image might still be RGBA transparent.
        If SD generated it, it's RGB with likely a 'clean white background' generated by SD.
        
        To rely on specific hex color, we should run Rembg ONE LAST TIME on the generated output?
        Because SD might generate a 'clean background' but not the EXACT hex code we want.
        """
        print(f"Running Step D: Post-processing with color {bg_color_hex}...")
        
        # Convert hex to RGB
        bg_color_hex = bg_color_hex.lstrip('#')
        bg_rgb = tuple(int(bg_color_hex[i:i+2], 16) for i in (0, 2, 4))
        
        # If image is RGBA (from preserve_outfit path), use its alpha.
        if image.mode == "RGBA":
            foreground = image
        else:
            # If RGB (from SD), we need to remove the generated background to place our custom color.
            # SD was prompted for "clean white background", so rembg should work well.
            foreground = rembg.remove(image, session=self.rembg_session)
            
        # Create solid color background
        final_bg = Image.new("RGB", foreground.size, bg_rgb)
        final_bg.paste(foreground, mask=foreground.split()[3])
        
        return final_bg
