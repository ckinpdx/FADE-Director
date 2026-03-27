import math
import torch


class ImageBatchChangeFPS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch tensor (B, H, W, C) representing the source frames."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "The frame rate at which the input batch was captured or generated.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "The desired output frame rate. Must be lower than source_fps to downsample.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("images", "fps", "frame_count")
    FUNCTION = "change_fps"
    CATEGORY = "image/animation"
    DESCRIPTION = (
        "Downsamples an IMAGE batch from source_fps to target_fps by dropping frames "
        "using nearest-neighbor (floor-based) index selection. No interpolation is "
        "performed. If target_fps >= source_fps the batch is passed through unchanged. "
        "Output frame count uses ceil to preserve the final frame, ensuring LTX 8k+1 "
        "frame counts remain valid for downstream models requiring 4k+1 (e.g. Wan/HuMo)."
    )

    def change_fps(self, images, source_fps, target_fps):
        num_frames = images.shape[0]

        if target_fps >= source_fps:
            print(
                f"[ComfyUI-FPSChange] Warning: target_fps ({target_fps}) >= source_fps "
                f"({source_fps}). Passing frames through unchanged."
            )
            return (images, target_fps, num_frames)

        if num_frames <= 1:
            return (images, target_fps, num_frames)

        duration = num_frames / source_fps
        # ceil preserves the final frame and ensures 8k+1 inputs produce 4k+1 outputs
        # when halving (e.g. 50fps → 25fps), keeping frame counts valid for Wan/HuMo.
        output_count = max(1, math.ceil(duration * target_fps))

        indices = [
            min(math.floor(i * source_fps / target_fps), num_frames - 1)
            for i in range(output_count)
        ]

        output_images = images[indices]
        print(f"[ComfyUI-FPSChange] {num_frames} frames @ {source_fps}fps → {output_count} frames @ {target_fps}fps")
        return (output_images, target_fps, output_count)


NODE_CLASS_MAPPINGS = {
    "ImageBatchChangeFPS": ImageBatchChangeFPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchChangeFPS": "Change Image Batch FPS",
}
