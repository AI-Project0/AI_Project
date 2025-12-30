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
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

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
            max_num_faces=5, # Increased to detect multiple faces for safety checks
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Lazy load HD models
        self.upscaler = None
        self.face_restorer = None
        
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
                safety_checker=None, # Disable for speed and avoiding false positives on harmless ID photos
                use_safetensors=True, # Force safe loading to bypass torch.load vulnerability
                variant="fp16"        # Use fp16 weights
            )
            
            # Use DPMSolver for faster inference (20-30 steps)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            if self.device == "cuda":
                # For low VRAM GPUs (like RTX 3050 4GB), use model_cpu_offload 
                # to only keep active components in VRAM. This prevents spilling into system RAM.
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing() # Further optimize memory
            else:
                self.pipe.to(self.device)
            
            print(f"Models loaded successfully onto {self.device} (CPU offload enabled if CUDA).")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e

    def _load_hd_models(self):
        """Lazy load Real-ESRGAN and GFPGAN models."""
        if self.upscaler is not None and self.face_restorer is not None:
            return

        print("Loading HD Restoration models...")
        try:
            # 1. Initialize Real-ESRGAN (x4 plus)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(self.device == "cuda"),
                device=self.device
            )

            # 2. Initialize GFPGAN (Face Restoration)
            self.face_restorer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                device=self.device
            )
            print("HD Restoration models loaded.")
        except Exception as e:
            print(f"Error loading HD models: {e}")
            # We don't raise here to allow the main process to continue even if HD fails

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
            
        if self._detect_jewelry(image, face_landmarks):
            raise ValueError("偵測到耳環或飾品，請移除後重新拍攝以符合正式證件照規範")

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
            # Step B: Masking (Face Protection Mask + Head Overlay Mask)
            mask_img, head_overlay_mask = self._step_b_masking(processed_img, face_landmarks)
            
            # Step C: Generation (SD Inpainting + Original Face Overlay)
            generated_img = self._step_c_generation(processed_img, mask_img, head_overlay_mask, prompt, negative_prompt, bg_color)
        
        # Step D: Post-processing (Final Composition with specific BG color)
        final_img = self._step_d_postprocessing(generated_img, bg_color)
        
        # Step E: HD Enhancement (Face Restoration + 4x Upscale)
        try:
            final_img = self._hd_enhancement(final_img)
        except Exception as e:
            print(f"Warning: HD Enhancement failed: {e}")
        
        # Return bytes
        output_buffer = io.BytesIO()
        final_img.save(output_buffer, format="PNG", quality=95)
        
        # Cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return output_buffer.getvalue()

    def _step_a_preprocessing(self, image: Image.Image, aspect_ratio: Tuple[int, int] = (35, 45)) -> Tuple[Image.Image, List]:
        """
        Removes background and crops image to target ratio centered on face.
        Returns: RGBA Image (Transparent Background) - Critical for later compositing.
        """
        print(f"Running Step A: Preprocessing with ratio {aspect_ratio}...")
        
        # 1. Remove background
        no_bg_image = rembg.remove(image, session=self.rembg_session)
        # --- Balanced Alpha Scrub (Step A) ---
        # Clean background residue WITHOUT cutting into head anatomy.
        rgba_np = np.array(no_bg_image)
        a_channel = rgba_np[:, :, 3]
        kernel = np.ones((3, 3), np.uint8)
        # 1 iteration is usually enough to kill the rim-halo
        rgba_np[:, :, 3] = cv2.erode(a_channel, kernel, iterations=1)
        no_bg_image = Image.fromarray(rgba_np)
        
        # For landmarks, we need an RGB image (MediaPipe doesn't like Alpha sometimes or needs contrast)
        # Composite against white for detection
        background = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        background.paste(no_bg_image, mask=no_bg_image.split()[3])
        rgb_for_detection = background
        
        # 2. Detect Face Landmarks
        img_np = np.array(rgb_for_detection)
        results = self.face_mesh.process(img_np)
        
        landmarks_list = results.multi_face_landmarks
        if not landmarks_list:
            return no_bg_image, []
            
        # Multi-Face Guardrail: Return error if more than 1 face detected
        if len(landmarks_list) > 1:
            raise ValueError("偵測到多張人臉，證件照僅支援單人，請重新上傳")
            
        landmarks = landmarks_list[0].landmark
        
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
        target_r = aspect_ratio[0] / aspect_ratio[1]
        
        # Crop Height logic: Face should be ~45-50% of height
        target_crop_h = face_height * 2.2 
        target_crop_w = target_crop_h * target_r
        
        # Center horizontally, Offset vertically
        # FIX: Ensure more headroom by shifting the crop down slightly or reducing top offset
        # Original: top = face_center_y - (target_crop_h * 0.45) 
        # Refined: Shift the window up (decrease top) to include more hair/top background
        top = face_center_y - (target_crop_h * 0.52) # 0.52 instead of 0.45 to move the crop 'up' (showing more headroom)
        left = face_center_x - (target_crop_w / 2)
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

    def _step_b_masking(self, image: Image.Image, landmarks: List) -> Tuple[Image.Image, Image.Image]:
        """
        Creates masks for 3-Stage Logic:
        1. Inpainting Mask: Protects Head, Repaints Body + Background.
        2. Head Overlay Mask: Isolates Original Head for synthesis.
        """
        print("Running Step B: Masking (Revised for Body Inpainting)...")
        w, h = image.size
        
        # 1. Get Head/Hair Mask
        # Step A returns RGBA where alpha is the person.
        full_person_mask = image.split()[3]
        
        # Create a Copy for the Head Isolation
        head_only_mask = full_person_mask.copy()
        draw = ImageDraw.Draw(head_only_mask)
        
        if landmarks:
            # Chin landmark (152) defines the boundary
            chin = landmarks[152]
            # Cut off at chin + 2px margin to protect jawline but allow neck repaint
            # This is critical for the suit collar to connect naturally.
            cutoff_y = int(chin.y * h) + 2
            draw.rectangle([(0, cutoff_y), (w, h)], fill=0)
        
        # Soften the protection zone for seamless blending
        # No more dilation - it causes "floating/halo" gaps.
        head_only_mask = head_only_mask.filter(ImageFilter.GaussianBlur(radius=5))

        # 2. SEAMLESS INPAINTING MASK (SD Input)
        # White = Repaint (Everything except head), Black = Protect (Head)
        inpainting_mask = Image.new("L", (w, h), 255)
        inpainting_mask.paste(0, mask=head_only_mask)
        
        # Soften the transition for SD
        inpainting_mask = inpainting_mask.filter(ImageFilter.GaussianBlur(radius=7)) # Tuned for precise boundaries

        return inpainting_mask, head_only_mask

    def _step_c_generation(
        self, 
        image: Image.Image, 
        mask: Image.Image, # Inpainting Mask
        head_overlay_mask: Image.Image, # Binary Head Isolation
        prompt: str, 
        negative_prompt: str,
        bg_color_hex: str = "#FFFFFF"
    ) -> Image.Image:
        """
        Stage 2 & 3: SD Inpainting + Advanced Alpha Blending Overlay
        """
        print(f"Running Step C: Generation... Structural Anchor Strategy (BG: {bg_color_hex}).")
        
        # --- Stage 2: SD Generation (Body & BG) ---
        # "STRUCTURAL ANCHOR" STRATEGY:
        # Instead of pure white, we use a blurred version of the background.
        # This gives SD "human shape" anchors (shoulders) to work with.
        
        # Convert hex to RGB for the base canvas
        bg_color_hex_clean = bg_color_hex.lstrip('#')
        bg_rgb = tuple(int(bg_color_hex_clean[i:i+2], 16) for i in (0, 2, 4))
        
        # Prepare Base Image: ORGANIC STRUCTURAL ANCHOR
        # We use the original person on target BG, but heavily blurred to kill clothing patterns.
        base_bg = Image.new("RGB", image.size, bg_rgb)
        # Composite person on target bg to ground lighting
        person_on_bg = Image.composite(image.convert("RGB"), base_bg, image.split()[3])
        # Heavier blur (radius 15) completely kills pajama textures for SD
        base_image = person_on_bg.filter(ImageFilter.GaussianBlur(radius=15))
        # Protect the crisp head area from being blurred
        base_image.paste(image, mask=head_overlay_mask)

        # Mapping prompt color
        if bg_color_hex_clean.upper() == "FFFFFF": color_name = "white"
        elif bg_color_hex_clean.upper() in ["4B89DC", "0000FF", "007BFF"]: color_name = "blue"
        elif bg_color_hex_clean.upper() in ["494949", "808080", "A9A9A9"]: color_name = "grey"
        else: color_name = "solid"

        # Optimized Prompt
        base_prompt = f"(wearing professional formal dark business suit with lapels and collar:1.4), (wearing crisp white collared shirt:1.2), ({color_name} background:1.3), (natural ears:1.2), (best quality, masterpiece, hyperrealism:1.3), professional ID photo, raw photo, fujifilm, dslr, (detailed skin texture, visible pores:1.1), studio lighting, soft shadows, sharp focus"
        final_prompt = f"{base_prompt}, ({prompt}:1.2)"
        
        # Stronger negative weights
        base_negative = "(earrings, jewelry, piercings:1.6), (t-shirt, round neck shirt:1.4), (3d render, cgi, cartoon, anime, drawing, painting:1.3), (plastic skin:1.5), bad anatomy, blurry, low quality"
        final_negative = f"{base_negative}, {negative_prompt}"
        
        generated_raw = self.pipe(
            prompt=final_prompt,
            negative_prompt=final_negative,
            image=base_image,
            mask_image=mask,
            num_inference_steps=40,
            guidance_scale=8.0,
            strength=0.98 # Force 100% transformation from pajamas to suit
        ).images[0]
        
        # --- Stage 3: ADVANCED FACE SYNTHESIS ---
        print("Applying Advanced Face Overlay (Edge Refinement)...")
        
        target_size = generated_raw.size
        w, h = target_size
        
        # Prepare Original Face
        original_face_rgba = image.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
        face_np = np.array(original_face_rgba)
        
        # 2. Prepare the main Head Overlay Mask
        mask_np = np.array(head_overlay_mask.resize(target_size))
        
        # --- Stage 3.2: Professional Alpha Blending ---
        # Erode minimally (1 iteration) to preserve hair details while cleaning edges
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask_np, kernel, iterations=1)
        
        # Gaussian Blur for soft edges
        blurred_mask = cv2.GaussianBlur(eroded_mask, (15, 15), 0)
        final_alpha_mask = Image.fromarray(blurred_mask).convert("L")
        
        # 3. Final Composite
        original_face_processed = Image.fromarray(face_np)
        background_canvas = generated_raw.convert("RGBA")
        background_canvas.paste(original_face_processed, (0, 0), mask=final_alpha_mask)
        
        return background_canvas.convert("RGB")

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
            # If the image was generated directly on the target color in Step C, 
            # we should check if we actually need rembg again. 
            # Generating direct avoids "rembg on white suit" failures.
            # For now, let's keep rembg but rely on the fact that Step C 
            # now uses the correct background color, which might make it 'clean' enough.
            # However, SD adds texture, so rembg is still useful to "flatten" the background.
            foreground = rembg.remove(image, session=self.rembg_session)
            
            # --- Edge Smoothing Post-Rembg ---
            # Extract alpha, smooth it, and put it back to avoid jagged artifacts
            f_rgba = np.array(foreground)
            f_alpha = f_rgba[:, :, 3]
            # Blur only (NO Dilation) to clean up "staircase" edges from rembg
            # Dilating alpha causes "glowing/bleeding" halos.
            f_alpha = cv2.GaussianBlur(f_alpha, (5, 5), 0)
            f_rgba[:, :, 3] = f_alpha
            foreground = Image.fromarray(f_rgba)
            
        # Create solid color background
        final_bg = Image.new("RGB", foreground.size, bg_rgb)
        
        # Apply Smart Drop Shadow
        final_canvas = self._apply_smart_drop_shadow(final_bg, foreground)
        
        return final_canvas

    def _apply_smart_drop_shadow(self, background: Image.Image, foreground: Image.Image) -> Image.Image:
        """
        Creates a photographic depth effect by adding a soft, offset shadow.
        """
        print("Applying Smart Drop Shadow...")
        
        # 1. Create a shadow layer from the foreground's alpha
        alpha = foreground.split()[3]
        
        # 2. Build the shadow (Reduced Opacity to avoid halos on blue)
        shadow_opacity = 60 # 0-255 scale (approx 23%)
        shadow = Image.new("L", foreground.size, 0)
        shadow_mask = alpha.point(lambda p: shadow_opacity if p > 0 else 0)
        
        # 3. Apply heavy Gaussian Blur to the shadow mask
        shadow_blur_radius = 15
        blurred_shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=shadow_blur_radius))
        
        # 4. Create the shadow layer
        shadow_layer = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
        # Use black color (0,0,0) with the blurred mask as the alpha channel
        black_img = Image.new("RGBA", foreground.size, (0, 0, 0, 255))
        shadow_layer.paste(black_img, mask=blurred_shadow_mask)
        
        # 5. Offset the shadow (Slightly to the bottom-right for studio look)
        offset_x = 8
        offset_y = 12
        offset_shadow = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
        offset_shadow.paste(shadow_layer, (offset_x, offset_y))
        
        # 6. Composite Everything
        # Background -> Shadow -> Foreground
        result = background.convert("RGBA")
        result.alpha_composite(offset_shadow)
        result.alpha_composite(foreground)
        
        return result.convert("RGB")

    def _detect_jewelry(self, image: Image.Image, landmarks: List) -> bool:
        """
        Heuristic to detect earrings/jewelry using earlobe landmarks.
        Checks for high color variance or metallic brightness in the ear area.
        """
        if not landmarks: return False
        try:
            w, h = image.size
            # 177: Left Earlobe, 406: Right Earlobe
            ears = [landmarks[177], landmarks[406]]
            
            roi_size = int(w * 0.04)
            img_np = np.array(image.convert("RGB"))
            
            for ear in ears:
                x, y = int(ear.x * w), int(ear.y * h)
                x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
                x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
                
                if x2 <= x1 or y2 <= y1: continue
                roi = img_np[y1:y2, x1:x2]
                
                # --- Multi-factor Jewelry Scoring ---
                score = 0
                
                # 1. Metallic Glint Detection (High Brightness Cluster)
                roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
                l_channel = roi_lab[:, :, 0]
                # High brightness pixels (near white glints)
                bright_peaks = np.count_nonzero(l_channel > 248)
                if bright_peaks > (roi.size * 0.005): # At least 0.5% area is glinting
                    score += 2
                elif np.max(l_channel) > 240:
                    score += 1
                
                # 2. Structural Complexity (Sharp Edges + Variance)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                # Find sharp edges (jewelry has harder edges than hair)
                edges = cv2.Canny(gray_roi, 100, 200)
                edge_density = np.count_nonzero(edges) / (roi.size / 3) # normalized
                
                variance = np.var(gray_roi)
                
                if variance > 3000 and edge_density > 0.12:
                    score += 2
                elif variance > 1800:
                    score += 1
                
                # Flag if score is high enough (Threshold = 3)
                if score >= 3:
                     return True
            return False
        except:
            return False

    def analyze_face(self, image_bytes: bytes, target_ratio: float = 0.8) -> dict:
        """
        Calculates the recommended crop box using specific ID photo landmarks.
        Refined Logic:
        - Top: Landmark 10
        - Chin: Landmark 152
        - Nose: Landmark 4 (Center)
        Returns: {x: %, y: %, width: %, height: %} (Relative 0-100)
        """
        print(f"Analyzing face with target_ratio: {target_ratio}...")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = image.size
            
            # Detect Face
            img_np = np.array(image)
            results = self.face_mesh.process(img_np)
            
            if not results.multi_face_landmarks:
                # Default behavior: centered crop if no face
                crop_h = 0.8 # 80% of height
                crop_w = crop_h * target_ratio
                return {
                    "x": (0.5 - crop_w / 2) * 100,
                    "y": (0.5 - crop_h / 2) * 100,
                    "width": crop_w * 100,
                    "height": crop_h * 100
                }
                
            # Multi-Face Guardrail
            if len(results.multi_face_landmarks) > 1:
                raise ValueError("偵測到多張人臉，證件照僅支援單人，請重新上傳")
                
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. Get Key Points (Relative 0.0 - 1.0)
            top = landmarks[10]
            chin = landmarks[152]
            nose = landmarks[4]
            
            # 2. Calculate Vertical Bounds
            face_height = chin.y - top.y
            
            # Headroom (0.6x face height)
            upper_y = top.y - (face_height * 0.6)
            # Lower bound (Chin + 0.5x face height)
            lower_y = chin.y + (face_height * 0.5)
            
            crop_h = lower_y - upper_y
            crop_w = crop_h * target_ratio
            
            # 3. Horizontal Center (Centered on Nose)
            upper_x = nose.x - (crop_w / 2)
            
            # 4. Final Clamping & Formatting (0-100)
            return {
                "x": max(0, min(upper_x, 1)) * 100,
                "y": max(0, min(upper_y, 1)) * 100,
                "width": max(0, min(crop_w, 1)) * 100,
                "height": max(0, min(crop_h, 1)) * 100
            }
        except Exception as e:
            print(f"Error in analyze_face: {e}")
            return None

    def _hd_enhancement(self, pil_image: Image.Image) -> Image.Image:
        """
        Applies Face Restoration (GFPGAN) and then 4x Super-Resolution (Real-ESRGAN).
        """
        print("Starting HD Enhancement pipeline...")
        self._load_hd_models()
        
        if self.upscaler is None or self.face_restorer is None:
            print("HD models not loaded, skipping enhancement.")
            return pil_image
            
        # Convert PIL to BGR OpenCV
        img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 1. Face Restoration (GFPGAN)
        # Note: GFPGAN expects BGR and returns BGR
        # Reduced fidelity weight (effectively blending) happens by mixing original back
        _, _, restored_img_bgr = self.face_restorer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
        
        # Blending restored face with original to keep skin texture (40% AI / 60% Original)
        # This prevents the "Plastic/Wax figure" look
        alpha = 0.30  # Balanced fidelity (30% AI / 70% Original)
        enhanced_bgr = cv2.addWeighted(restored_img_bgr, alpha, img_np, 1 - alpha, 0)
        
        # 2. Real-ESRGAN Upscaling (4x)
        # Note: can handle BGR directly
        upscaled_img, _ = self.upscaler.enhance(enhanced_bgr, outscale=4)
        
        # Convert back to PIL RGB
        return Image.fromarray(cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB))

    def _detect_jewelry(self, image: Image.Image, landmarks: List) -> bool:
        """
        Heuristic to detect earrings/jewelry using earlobe landmarks.
        Checks for high color variance or metallic brightness in the ear area.
        """
        if not landmarks: return False
        try:
            w, h = image.size
            # 177: Left Earlobe, 406: Right Earlobe
            ears = [landmarks[177], landmarks[406]]
            
            roi_size = int(w * 0.04)
            img_np = np.array(image.convert("RGB"))
            
            for ear in ears:
                x, y = int(ear.x * w), int(ear.y * h)
                x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
                x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
                
                if x2 <= x1 or y2 <= y1: continue
                roi = img_np[y1:y2, x1:x2]
                
                # --- Multi-factor Jewelry Scoring ---
                score = 0
                
                # 1. Metallic Glint Detection (High Brightness Cluster)
                roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
                l_channel = roi_lab[:, :, 0]
                # High brightness pixels (near white glints)
                bright_peaks = np.count_nonzero(l_channel > 248)
                if bright_peaks > (roi.size * 0.005): # At least 0.5% area is glinting
                    score += 2
                elif np.max(l_channel) > 240:
                    score += 1
                
                # 2. Structural Complexity (Sharp Edges + Variance)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                # Find sharp edges (jewelry has harder edges than hair)
                edges = cv2.Canny(gray_roi, 100, 200)
                edge_density = np.count_nonzero(edges) / (roi.size / 3) # normalized
                
                variance = np.var(gray_roi)
                
                if variance > 3000 and edge_density > 0.12:
                    score += 2
                elif variance > 1800:
                    score += 1
                
                # Flag if score is high enough (Threshold = 3)
                if score >= 3:
                     return True
            return False
        except:
            return False
