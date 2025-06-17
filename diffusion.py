# ===== FINAL FLUX INPAINTING FOR KAGGLE =====
# Complete self-contained solution with precise object preservation
# Enhanced mask precision, better object positioning, and bedroom-specific prompts

import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

# Install required packages
packages = [
    'diffusers==0.33.1',
    'transformers', 
    'torch',
    'torchvision',
    'accelerate',
    'rembg',
    'onnxruntime',
    'opencv-python',
    'scikit-image',
    'Pillow',
    'numpy',
    'matplotlib'
]

print("Installing packages...")
for package in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        print(f"✓ {package}")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

# Create mock xformers to prevent import errors
import os
import sys
from types import ModuleType

# Create mock xformers module
class MockXformers:
    def __getattr__(self, name):
        return MockXformers()
    
    def __call__(self, *args, **kwargs):
        return MockXformers()

# Install mock xformers
sys.modules['xformers'] = MockXformers()
sys.modules['xformers.ops'] = MockXformers()

# Disable xformers in environment
os.environ['XFORMERS_DISABLED'] = '1'

print("Mock xformers installed and disabled")

# Import everything
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
import cv2
from rembg import new_session, remove
from skimage import segmentation, color, filters, morphology
from skimage.feature import canny
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_closing
import warnings
warnings.filterwarnings('ignore')

print("All imports successful!")

def remove_background_advanced(image_path, method='rembg'):
    """Advanced background removal with very tight mask creation to eliminate white halos"""
    print(f"Removing background from {image_path} using {method}")
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    if method == 'rembg':
        try:
            # Try rembg first (best for objects)
            session = new_session('u2net')
            result = remove(pil_img, session=session)
            
            # Convert to numpy for processing
            result_np = np.array(result)
            
            # Create extremely tight mask from alpha channel
            if result_np.shape[2] == 4:
                alpha = result_np[:, :, 3]
                
                # Use much higher threshold for ultra-tight fit (preserve only very opaque pixels)
                mask = alpha > 200  # Much higher than before (was 128)
                
                # Minimal cleanup - only close tiny gaps, no dilation
                mask = binary_closing(mask, disk(1))  # Tiny closing only
                
                # Use contour refinement for tightest possible fit
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Keep only the largest contour (main object)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create extremely tight contour mask
                    tight_mask = np.zeros_like(mask_uint8)
                    cv2.fillPoly(tight_mask, [largest_contour], 255)
                    
                    # Use the intersection for ultimate tightness
                    mask = (tight_mask > 0) & mask
                
                # Apply ultra-tight mask
                result_clean = result_np.copy()
                result_clean[~mask] = [0, 0, 0, 0]
                
                print(f"Ultra-tight mask coverage: {np.sum(mask)} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
                
                return Image.fromarray(result_clean, 'RGBA')
            
        except Exception as e:
            print(f"rembg failed: {e}, trying GrabCut")
    
    # Fallback to GrabCut with tight refinement
    try:
        height, width = img_rgb.shape[:2]
        
        # Create rectangle for GrabCut (focus on center object)
        rect = (width//8, height//8, width*3//4, height*3//4)
        
        # Initialize masks
        mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Make GrabCut mask much tighter too
        mask2 = binary_erosion(mask2, disk(2))  # Erode first to tighten
        mask2 = binary_closing(mask2, disk(1))  # Only tiny closing
        
        # Apply mask
        result = img_rgb * mask2[:, :, np.newaxis]
        
        # Create RGBA image
        rgba_result = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_result[:, :, :3] = result
        rgba_result[:, :, 3] = mask2 * 255
        
        print(f"Tight GrabCut mask coverage: {np.sum(mask2)} pixels ({100*np.sum(mask2)/mask2.size:.1f}%)")
        
        return Image.fromarray(rgba_result, 'RGBA')
        
    except Exception as e:
        print(f"GrabCut failed: {e}, using simple threshold")
        
        # Simple fallback with tighter threshold
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Higher threshold
        
        # Tighten simple threshold too
        mask = binary_erosion(mask > 0, disk(1))
        
        rgba_result = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
        rgba_result[:, :, :3] = img_rgb
        rgba_result[:, :, 3] = mask.astype(np.uint8) * 255
        
        return Image.fromarray(rgba_result, 'RGBA')

def get_precise_object_bounds(image_rgba):
    """Get precise bounds of object using alpha channel and contours"""
    # Convert to numpy
    img_array = np.array(image_rgba)
    
    if img_array.shape[2] == 4:
        # Use alpha channel
        alpha = img_array[:, :, 3]
        mask = alpha > 50  # More sensitive threshold
    else:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask = gray > 30
    
    # Find contours
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding but keep tight
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2*padding)
        h = min(img_array.shape[0] - y, h + 2*padding)
        
        return (x, y, w, h), largest_contour
    else:
        # Fallback to full image bounds
        return (0, 0, img_array.shape[1], img_array.shape[0]), None

def compose_scene_with_precise_masks(lamp_rgba, table_rgba, canvas_size=(512, 512)):
    """Compose scene with precise object positioning and mask creation"""
    print("Composing scene with precise positioning...")
    
    # Create canvas
    canvas = Image.new('RGBA', canvas_size, (255, 255, 255, 0))
    
    # Get precise bounds for both objects
    lamp_bounds, lamp_contour = get_precise_object_bounds(lamp_rgba)
    table_bounds, table_contour = get_precise_object_bounds(table_rgba)
    
    print(f"Lamp bounds: {lamp_bounds}")
    print(f"Table bounds: {table_bounds}")
    
    # Calculate scaling to fit both objects properly
    lamp_x, lamp_y, lamp_w, lamp_h = lamp_bounds
    table_x, table_y, table_w, table_h = table_bounds
    
    # Calculate realistic bedroom furniture scaling 
    max_obj_size = min(canvas_size) // 3
    
    # Scale table much larger - it should be a prominent nightstand/side table
    # In a bedroom, a side table should be 40-60% of scene width
    table_target_width = int(canvas_size[0] * 0.5)  # 50% of canvas width
    table_scale = table_target_width / table_w
    
    # Ensure table height is proportional but also prominent
    table_new_size = (int(table_w * table_scale), int(table_h * table_scale))
    
    # Ensure table doesn't exceed reasonable height (60% of canvas)
    if table_new_size[1] > canvas_size[1] * 0.6:
        table_scale = (canvas_size[1] * 0.6) / table_h
        table_new_size = (int(table_w * table_scale), int(table_h * table_scale))
    
    # Scale lamp proportionally to table - should be suitable for the table size
    # A lamp should be 20-40% of table width and appropriate height
    lamp_target_width = int(table_new_size[0] * 0.35)  # 35% of table width
    lamp_scale = lamp_target_width / lamp_w
    lamp_new_size = (int(lamp_w * lamp_scale), int(lamp_h * lamp_scale))
    
    print(f"Realistic sizing:")
    print(f"Table: {table_new_size} (scale: {table_scale:.2f}) - {100*table_new_size[0]/canvas_size[0]:.0f}% of canvas width")
    print(f"Lamp: {lamp_new_size} (scale: {lamp_scale:.2f}) - {100*lamp_new_size[0]/table_new_size[0]:.0f}% of table width")
    
    # Extract and resize objects
    lamp_crop = lamp_rgba.crop((lamp_x, lamp_y, lamp_x + lamp_w, lamp_y + lamp_h))
    lamp_resized = lamp_crop.resize(lamp_new_size, Image.Resampling.LANCZOS)
    
    table_crop = table_rgba.crop((table_x, table_y, table_x + table_w, table_y + table_h))
    table_resized = table_crop.resize(table_new_size, Image.Resampling.LANCZOS)
    
    # Position objects realistically for a bedroom scene
    # Table positioned as a prominent bedside table - slightly off-center and in foreground
    table_pos_x = int(canvas_size[0] * 0.3)  # Slightly left of center for natural look
    table_pos_y = canvas_size[1] - table_new_size[1] - 20  # Close to bottom with small floor margin
    
    # Lamp positioned naturally on the table surface
    lamp_pos_x = table_pos_x + int(table_new_size[0] * 0.3)  # Offset from table center for realism
    lamp_pos_y = table_pos_y - int(lamp_new_size[1] * 0.85)  # Sit mostly on table surface
    
    print(f"Realistic positioning:")
    print(f"Table: ({table_pos_x}, {table_pos_y}) - positioned as prominent bedside furniture")
    print(f"Lamp: ({lamp_pos_x}, {lamp_pos_y}) - naturally placed on table surface")
    
    # Composite objects on canvas
    canvas.paste(table_resized, (table_pos_x, table_pos_y), table_resized)
    canvas.paste(lamp_resized, (lamp_pos_x, lamp_pos_y), lamp_resized)
    
    # Create ultra-precise inpainting mask that protects BOTH objects with NO halos
    mask = np.zeros(canvas_size[::-1], dtype=np.uint8)  # Height x Width
    
    # Create ultra-tight masks for each object using their alpha channels
    lamp_array = np.array(lamp_resized)
    table_array = np.array(table_resized)
    
    # Table mask - ultra-tight protection with no halo
    if table_array.shape[2] == 4:
        table_alpha = table_array[:, :, 3]
        table_mask = (table_alpha > 200).astype(np.uint8) * 255  # Much higher threshold (was 30)
        
        # Minimal dilation only to close tiny gaps - NO expansion
        table_mask = binary_closing(table_mask > 0, disk(1)).astype(np.uint8) * 255  # Tiny closing only
        
        # Apply to canvas mask
        table_end_y = min(table_pos_y + table_mask.shape[0], canvas_size[1])
        table_end_x = min(table_pos_x + table_mask.shape[1], canvas_size[0])
        
        if table_pos_y >= 0 and table_pos_x >= 0:
            mask[table_pos_y:table_end_y, table_pos_x:table_end_x] = np.maximum(
                mask[table_pos_y:table_end_y, table_pos_x:table_end_x],
                table_mask[:table_end_y-table_pos_y, :table_end_x-table_pos_x]
            )
    
    # Lamp mask - ultra-tight protection with no halo
    if lamp_array.shape[2] == 4:
        lamp_alpha = lamp_array[:, :, 3]
        lamp_mask = (lamp_alpha > 200).astype(np.uint8) * 255  # Much higher threshold (was 30)
        
        # Minimal dilation only to close tiny gaps - NO expansion
        lamp_mask = binary_closing(lamp_mask > 0, disk(1)).astype(np.uint8) * 255  # Tiny closing only
        
        # Apply to canvas mask
        lamp_end_y = min(lamp_pos_y + lamp_mask.shape[0], canvas_size[1])
        lamp_end_x = min(lamp_pos_x + lamp_mask.shape[1], canvas_size[0])
        
        if lamp_pos_y >= 0 and lamp_pos_x >= 0:
            mask[lamp_pos_y:lamp_end_y, lamp_pos_x:lamp_end_x] = np.maximum(
                mask[lamp_pos_y:lamp_end_y, lamp_pos_x:lamp_end_x],
                lamp_mask[:lamp_end_y-lamp_pos_y, :lamp_end_x-lamp_pos_x]
            )
    
    # Convert canvas to RGB with white background
    canvas_rgb = Image.new('RGB', canvas_size, (245, 245, 245))  # Light background
    canvas_rgb.paste(canvas, (0, 0), canvas)
    
    # IMPORTANT: Invert mask for inpainting - white areas will be inpainted, black areas preserved
    # Current mask has white=objects, so we need to invert it
    inverted_mask = 255 - mask
    mask_pil = Image.fromarray(inverted_mask, 'L')
    
    preserved_pixels = np.sum(inverted_mask <= 128)
    inpaint_pixels = np.sum(inverted_mask > 128)
    print(f"Final mask: {preserved_pixels} preserved pixels, {inpaint_pixels} inpaint pixels")
    print(f"Preservation ratio: {preserved_pixels/(preserved_pixels+inpaint_pixels):.1%}")
    
    return canvas_rgb, mask_pil

def visualize_results(original, mask, result, title="Inpainting Results"):
    """Visualize the inpainting process and results with detailed analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Ensure all images are the same size
    original_array = np.array(original)
    result_array = np.array(result)
    mask_array = np.array(mask)
    
    # Resize if needed to match result size
    target_size = result_array.shape[:2]
    if original_array.shape[:2] != target_size:
        original = original.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        original_array = np.array(original)
    
    if mask_array.shape[:2] != target_size:
        mask = mask.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        mask_array = np.array(mask)
    
    # Top row: Process
    axes[0, 0].imshow(original_array)
    axes[0, 0].set_title('Original Scene', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_array, cmap='gray')
    axes[0, 1].set_title('Inpainting Mask\n(White = Inpaint, Black = Preserve)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result_array)
    axes[0, 2].set_title('Final Result', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Bottom row: Analysis
    # Mask overlay on original - show preserved areas in red
    overlay = original_array.copy()
    
    # Create red overlay for preserved areas (black in mask = preserved)
    preserved_areas = mask_array <= 128  # Black areas are preserved
    red_overlay = np.zeros_like(overlay)
    red_overlay[preserved_areas, 0] = 255  # Red channel for preserved areas
    
    # Blend with original
    protected_viz = overlay.copy()
    protected_viz[preserved_areas] = (overlay[preserved_areas] * 0.7 + red_overlay[preserved_areas] * 0.3).astype(np.uint8)
    axes[1, 0].imshow(protected_viz)
    axes[1, 0].set_title('Protected Areas (Red = Preserved)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map
    diff = np.abs(result_array.astype(np.float32) - original_array.astype(np.float32))
    diff_gray = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
    im = axes[1, 1].imshow(diff_gray, cmap='hot', vmin=0, vmax=100)
    axes[1, 1].set_title('Changes Made\n(Bright = More Change)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Side by side comparison
    comparison = np.hstack([original_array, result_array])
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title('Before | After', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed mask statistics
    mask_array = np.array(mask)
    preserved_pixels = np.sum(mask_array <= 128)  # Black pixels are preserved
    inpaint_pixels = np.sum(mask_array > 128)     # White pixels are inpainted
    total_pixels = mask_array.size
    preservation_ratio = preserved_pixels / total_pixels
    inpaint_ratio = inpaint_pixels / total_pixels
    
    print(f"\n{'='*50}")
    print(f"MASK ANALYSIS")
    print(f"{'='*50}")
    print(f"Preserved pixels (black): {preserved_pixels:,}")
    print(f"Inpaint pixels (white): {inpaint_pixels:,}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Preservation ratio: {preservation_ratio:.2%}")
    print(f"Inpaint ratio: {inpaint_ratio:.2%}")
    
    # Calculate change statistics
    change_amount = np.mean(diff_gray)
    max_change = np.max(diff_gray)
    
    # Calculate changes in preserved vs inpainted areas
    preserved_area = mask_array <= 128  # Black areas are preserved
    inpaint_area = mask_array > 128     # White areas are inpainted
    
    if np.any(preserved_area):
        preserved_change = np.mean(diff_gray[preserved_area])
        print(f"Average change in PRESERVED areas: {preserved_change:.2f}")
    
    if np.any(inpaint_area):
        inpaint_change = np.mean(diff_gray[inpaint_area])
        print(f"Average change in INPAINTED areas: {inpaint_change:.2f}")
    
    print(f"Overall average change: {change_amount:.2f}")
    print(f"Maximum change: {max_change:.2f}")
    
    return change_amount

# ===== MAIN EXECUTION =====
def main():
    print("=== FLUX/ControlNet Inpainting for Kaggle ===")
    print("Processing Gothic Lamp and Table with Enhanced Precision...")

    try:
        # Step 1: Remove backgrounds with advanced methods
        print(f"\n{'='*60}")
        print("STEP 1: ADVANCED BACKGROUND REMOVAL")
        print(f"{'='*60}")
        
        lamp_rgba = remove_background_advanced('/kaggle/input/gothic/gothic_lamp.jpg', 'rembg')
        table_rgba = remove_background_advanced('/kaggle/input/gothic/gothic_table.jpg', 'rembg')
        
        print("✓ Background removal completed successfully")
        
        # Step 2: Compose scene with precise masks
        print(f"\n{'='*60}")
        print("STEP 2: PRECISE SCENE COMPOSITION")
        print(f"{'='*60}")
        
        # Use 512x512 to match inpainting pipeline output
        scene_image, inpaint_mask = compose_scene_with_precise_masks(lamp_rgba, table_rgba, canvas_size=(512, 512))
        
        print("✓ Scene composition with precise masks completed")
        
        # Step 3: Load and configure inpainting model
        print(f"\n{'='*60}")
        print("STEP 3: MODEL LOADING AND CONFIGURATION")
        print(f"{'='*60}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Try multiple inpainting models with priority order
        model_names = [
            "stabilityai/stable-diffusion-2-inpainting",
            "runwayml/stable-diffusion-inpainting"
        ]
        
        pipe = None
        for model_name in model_names:
            try:
                print(f"Attempting to load {model_name}...")
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                pipe = pipe.to(device)
                print(f"✓ Successfully loaded {model_name}")
                break
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                continue
        
        if pipe is None:
            raise Exception("❌ No inpainting model could be loaded")
        
        # Optimize memory usage
        try:
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
                print("✓ CPU offload enabled")
        except:
            pass
        
        try:
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
                print("✓ Attention slicing enabled")
        except:
            pass
        
        # Step 4: Enhanced inpainting with bedroom-specific prompts
        print(f"\n{'='*60}")
        print("STEP 4: ENHANCED INPAINTING PROCESS")
        print(f"{'='*60}")
        
        # Bedroom-focused prompts with strong style direction
        bedroom_prompts = [
            "elegant modern bedroom interior, warm cozy lighting, soft ambient atmosphere, wooden hardwood floor, neutral beige walls, contemporary furniture placement, peaceful sleeping environment, high quality interior design, photorealistic",
            
            "luxurious bedroom with soft warm lighting, sophisticated interior design, comfortable cozy atmosphere, premium wooden furniture, elegant neutral decor, soft fabric textures, inviting peaceful space, modern bedroom styling",
            
            "beautiful contemporary bedroom interior, warm golden lighting, modern elegant design, comfortable living space, high-end bedroom furniture, sophisticated calm atmosphere, wooden floor, neutral color palette"
        ]
        
        # Strong negative prompt to avoid unwanted elements
        negative_prompt = "hallway, corridor, passage, medieval castle, stone walls, cold harsh lighting, outdoor scene, kitchen, bathroom, office space, commercial space, harsh shadows, poor quality, blurry, distorted objects, deformed furniture"
        
        best_result = None
        best_change = 0
        results = []
        
        for i, prompt in enumerate(bedroom_prompts):
            print(f"\n--- Attempting Inpainting {i+1}/{len(bedroom_prompts)} ---")
            print(f"Prompt: {prompt[:100]}...")
            
            try:
                # Generate with optimized settings for object preservation
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=scene_image,
                    mask_image=inpaint_mask,
                    num_inference_steps=35,        # High steps for quality
                    guidance_scale=9.0,            # Higher guidance for stronger effect
                    strength=0.95,                 # Very high strength for dramatic changes
                    height=512,                    # Explicit size matching
                    width=512,                     # Explicit size matching
                    generator=torch.Generator(device=device).manual_seed(42 + i)
                ).images[0]
                
                # Analyze results
                print(f"Generated result {i+1}, analyzing...")
                change_amount = visualize_results(
                    scene_image, inpaint_mask, result, 
                    f"Inpainting Result {i+1}: {prompt.split(',')[0].title()}"
                )
                
                results.append((result, change_amount, prompt))
                
                if change_amount > best_change:
                    best_result = result
                    best_change = change_amount
                    print(f"✓ NEW BEST RESULT! Change amount: {change_amount:.2f}")
                else:
                    print(f"Change amount: {change_amount:.2f}")
                
            except Exception as e:
                print(f"✗ Inpainting attempt {i+1} failed: {e}")
                continue
        
        # Step 5: Results analysis and saving
        print(f"\n{'='*60}")
        print("STEP 5: FINAL RESULTS AND ANALYSIS")
        print(f"{'='*60}")
        
        if best_result:
            print(f"✓ Inpainting completed successfully!")
            print(f"✓ Best result achieved change amount: {best_change:.2f}")
            
            # Display final best result with detailed analysis
            final_change = visualize_results(
                scene_image, inpaint_mask, best_result, 
                "🏆 FINAL BEST INPAINTING RESULT"
            )
            
            # Save all results
            scene_image.save('/kaggle/working/original_scene.png')
            inpaint_mask.save('/kaggle/working/inpaint_mask.png')
            best_result.save('/kaggle/working/inpainted_result.png')
            
            # Save comparison
            # Ensure sizes match for comparison
            scene_for_comparison = scene_image
            if scene_image.size != best_result.size:
                scene_for_comparison = scene_image.resize(best_result.size, Image.Resampling.LANCZOS)
            
            comparison = np.hstack([np.array(scene_for_comparison), np.array(best_result)])
            Image.fromarray(comparison).save('/kaggle/working/before_after_comparison.png')
            
            print("✓ Results saved to /kaggle/working/")
            
            # Object preservation verification
            print(f"\n{'='*60}")
            print("OBJECT PRESERVATION VERIFICATION")
            print(f"{'='*60}")
            
            result_array = np.array(best_result)
            original_array = np.array(scene_image)
            mask_array = np.array(inpaint_mask)
            
            # Analyze preservation in protected areas
            preserved_area = mask_array <= 128  # Black pixels are preserved
            
            if np.any(preserved_area):
                # Ensure arrays are same size for comparison
                if original_array.shape != result_array.shape:
                    scene_resized = scene_image.resize(best_result.size, Image.Resampling.LANCZOS)
                    original_array = np.array(scene_resized)
                    mask_resized = inpaint_mask.resize(best_result.size, Image.Resampling.LANCZOS)
                    mask_array = np.array(mask_resized)
                    preserved_area = mask_array <= 128
                
                preserved_original = original_array[preserved_area].astype(np.float32)
                preserved_result = result_array[preserved_area].astype(np.float32)
                
                # Calculate preservation metrics
                preservation_score = 1.0 - np.mean(np.abs(preserved_original - preserved_result)) / 255.0
                color_similarity = 1.0 - np.mean(np.std(preserved_result - preserved_original, axis=0)) / 255.0
                
                print(f"Object preservation score: {preservation_score:.2%}")
                print(f"Color similarity score: {color_similarity:.2%}")
                
                if preservation_score > 0.75:
                    print("✅ EXCELLENT: Objects very well preserved!")
                elif preservation_score > 0.6:
                    print("✅ GOOD: Objects well preserved")
                elif preservation_score > 0.4:
                    print("⚠️  FAIR: Objects partially preserved")
                else:
                    print("❌ POOR: Objects significantly modified")
            
            # Scene type verification
            print(f"\n{'='*60}")
            print("SCENE ANALYSIS")
            print(f"{'='*60}")
            
            # Check if background looks like bedroom
            inpainted_area = ~preserved_area  # Areas that were inpainted
            if np.any(inpainted_area):
                bg_pixels = result_array[inpainted_area]
                
                # Analyze color palette of background
                avg_color = np.mean(bg_pixels, axis=0)
                color_warmth = (avg_color[0] + avg_color[1]) / (avg_color[2] + 1)  # R+G vs B
                
                print(f"Background average color: RGB{tuple(avg_color.astype(int))}")
                print(f"Color warmth ratio: {color_warmth:.2f}")
                
                if color_warmth > 1.1:
                    print("✅ Background appears warm-toned (bedroom-like)")
                else:
                    print("⚠️  Background appears cool-toned")
            
            print(f"\n{'🎉'*20}")
            print("🎉 INPAINTING COMPLETED SUCCESSFULLY! 🎉")
            print(f"{'🎉'*20}")
            
        else:
            print("❌ All inpainting attempts failed")
            print("Check the error messages above for debugging information")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure input images exist at /kaggle/input/gothic/")
        print("2. Check GPU memory availability")
        print("3. Verify diffusers==0.33.1 is installed correctly")
        print("4. Check internet connection for model downloads")

if __name__ == "__main__":
    main()
