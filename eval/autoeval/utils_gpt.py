import asyncio
import base64
import cv2
from io import BytesIO
from itertools import islice
import json
import numpy as np
import openai
from openai import OpenAI
from openai import AsyncOpenAI
import os
from PIL import Image
import re
from tqdm.asyncio import tqdm_asyncio

import script_eval

def resize_to_side(image, target_size, fn="min"):
    width, height = image.size
    if fn == "min":
        denom = min(width, height)
    elif fn == "max":
        denom = max(width, height)
    scale_factor = target_size / denom
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def parse_json(output):
    try:
        return json.loads(output.replace("```json", "").replace("```", ""))
    except Exception as e:
        print("Parse JSON:", e)
        return None

def decode_base64_to_pil(base64_str):
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def pil_to_base64(image):
    # jpeg is faster to encode than png
    arr = np.array(image.convert("RGB"))
    _, buffer = cv2.imencode(".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")

def gpt_create_message(prompt, system_prompt=None, image_key=None, input_images=None, image_url=True, image_size=None):
    images = input_images
    if image_size is not None:
        images = [resize_to_side(image, image_size) for image in images]
    # handle multiple image keys
    if isinstance(image_key, (list, tuple)):
        parts = re.split("|".join(map(re.escape, image_key)), prompt)
    else:
        parts = prompt.split(image_key) if image_key is not None else [prompt]
    
    if images is not None and image_url:
        b64s = [pil_to_base64(img) for img in images]

    user_content = []
    for i, part in enumerate(parts):
        if part:
            user_content.append({"type": "text", "text": part})
        if images is not None and i < len(images):
            if image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64s[i]}"}
                })
            else:
                user_content.append({
                    "type": "image",
                    "image": images[i]
                })
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages

def qwen_create_message(prompt, system_prompt=None, image_key=None, input_images=None, output_images=None, image_url=True, image_size=None):
    # Process input images
    processed_input_images = []
    if input_images is not None:
        if image_size is not None:
            processed_input_images = [resize_to_side(image, image_size) for image in input_images]
        else:
            processed_input_images = input_images
    
    # Process output images  
    processed_output_images = []
    if output_images is not None:
        if image_size is not None:
            processed_output_images = [resize_to_side(image, image_size) for image in output_images]
        else:
            processed_output_images = output_images
    
    # Convert images to base64 if needed
    input_b64s = []
    output_b64s = []
    if image_url:
        if processed_input_images:
            input_b64s = [pil_to_base64(img) for img in processed_input_images]
        if processed_output_images:
            output_b64s = [pil_to_base64(img) for img in processed_output_images]

    # Split the prompt by the placeholders
    parts = prompt.split('<input_images>')
    if len(parts) == 2:
        before_input, after_input = parts
        # Split the after part by output_image placeholder
        middle_parts = after_input.split('<output_image>')
        if len(middle_parts) == 2:
            middle, after_output = middle_parts
        else:
            middle = after_input
            after_output = ""
    else:
        # No input_images placeholder, check for output_image only
        parts = prompt.split('<output_image>')
        if len(parts) == 2:
            before_input = parts[0]
            middle = ""
            after_output = parts[1]
        else:
            before_input = prompt
            middle = ""
            after_output = ""

    # Build the user content
    user_content = []
    
    # Add text before input_images
    if before_input.strip():
        user_content.append({"type": "text", "text": before_input})
    
    # Add input images
    if processed_input_images:
        for i, img in enumerate(processed_input_images):
            if image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{input_b64s[i]}"}
                })
            else:
                user_content.append({
                    "type": "image",
                    "image": img
                })
    
    # Add text between input_images and output_image
    if middle.strip():
        user_content.append({"type": "text", "text": middle})
    
    # Add output images
    if processed_output_images:
        for i, img in enumerate(processed_output_images):
            if image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{output_b64s[i]}"}
                })
            else:
                user_content.append({
                    "type": "image",
                    "image": img
                })
    
    # Add text after output_image
    if after_output.strip():
        user_content.append({"type": "text", "text": after_output})
    
    # Create messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    print("user content length", len(user_content))
    return messages

async def _gpt_batch_call_async(
    messages,
    max_concurrency=8,
    model="chatgpt-4o-latest",
    api_cls="AsyncOpenAI",
    **call_kwargs,
):
    api_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    async with getattr(openai, api_cls)(**api_kwargs) as client:
        sem = asyncio.Semaphore(max_concurrency)

        async def call(idx, m):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model, messages=m, **call_kwargs
                    )
                    return idx, resp.choices[0].message.content
            except Exception as e:
                print(e)
                return idx, None

        tasks = [asyncio.create_task(call(idx, m)) for idx, m in enumerate(messages)]
        # IMPORTANT: results need to return in order
        ordered = [None] * len(tasks)
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, content = await coro
            ordered[idx] = content
        return ordered
    
async def qwen_batch_call(
    messages,
    max_concurrency=2,
    api_cls="AsyncOpenAI",
    model="qwen/qwen2.5-vl-32b-instruct",
    **call_kwargs,
):
    max_concurrency=4
    api_kwargs = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    }
    client = getattr(openai, api_cls)(**api_kwargs)
    async with client:
        sem = asyncio.Semaphore(max_concurrency)

        async def call(idx, m):
            while True:
                tries = 0
                delay = 1
                try:
                    async with sem:
                        resp = await client.chat.completions.create(
                            model=model, messages=m
                        )
                        print(resp.choices[0].message.content)
                        return idx, resp.choices[0].message.content
                except Exception as e:
                    error_str = str(e).lower()
                    # Only retry if it's a rate limit error
                    if "rate" in error_str or "429" in error_str or "too many requests" in error_str:
                        print(f"[{idx}] Rate limit hit, retrying in {delay}s (attempt {tries + 1}/5): {e}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        tries += 1
                    else:
                        # For non-rate-limit errors, fail immediately
                        print(f"[{idx}] Non-rate-limit error, failing immediately: {e}")
                        return idx, None

        tasks = [asyncio.create_task(call(idx, m)) for idx, m in enumerate(messages)]
        # IMPORTANT: results need to return in order
        ordered = [None] * len(tasks)
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, content = await coro
            ordered[idx] = content
        return ordered

async def gemini_batch_call(
    messages,
    max_concurrency=2,
    api_cls="AsyncOpenAI",
    model="google/gemini-2.0-flash-lite-001",
    **call_kwargs,
):
    max_concurrency=2
    api_kwargs = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    }
    client = getattr(openai, api_cls)(**api_kwargs)
    async with client:
        sem = asyncio.Semaphore(max_concurrency)

        async def call(idx, m):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model, messages=m
                    )
                    raw_reply = resp.choices[0].message.content
                    score = script_eval.extract_score(raw_reply)
                    print(raw_reply)
                    return idx, raw_reply
            except Exception as e:
                print(e)
                return idx, None

        tasks = [asyncio.create_task(call(idx, m)) for idx, m in enumerate(messages)]
        # IMPORTANT: results need to return in order
        ordered = [None] * len(tasks)
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, content = await coro
            ordered[idx] = content
        return ordered

def gpt_batch_call(*args, **kwargs):
    coro = _gpt_batch_call_async(*args, **kwargs)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return coro