from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
replicate_api = os.getenv("REPLICATE_API_TOKEN")

import replicate

output = replicate.run(
    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "width": 1024,
        "height": 1024,
        "prompt": "An astronaut riding a rainbow unicorn",
        "refine": "no_refiner",
        "scheduler": "K_EULER",
        "lora_scale": 0.6,
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "apply_watermark": True,
        "high_noise_frac": 0.8,
        "negative_prompt": "",
        "prompt_strength": 0.8,
        "num_inference_steps": 50
    }
)
print(output)