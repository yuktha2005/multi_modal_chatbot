from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import os

# ---------- DEVICE ----------
device = "cpu"  # force CPU for i3 laptop

# ---------- IMAGE MODEL (BLIP) ----------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)

def generate_caption(image_path):
    """
    Generate a concise, cleaned-up caption for an image using BLIP
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Keep only the first sentence and remove repeated words
    first_sentence = caption.split(".")[0]
    words = first_sentence.split()
    seen = set()
    cleaned_words = []
    for w in words:
        if w.lower() not in seen:
            cleaned_words.append(w)
            seen.add(w.lower())
    cleaned_caption = " ".join(cleaned_words)

    return cleaned_caption


# ---------- TEXT MODEL (CPU-friendly GPT-Neo) ----------
llm_name = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    device_map={"": "cpu"},  # force CPU
    dtype=torch.float32
)

def generate_response(prompt, max_tokens=60):
    """
    Generate a concise text response from GPT-Neo
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the first sentence
    if "." in response:
        return response.split(".")[0].strip()
    elif "\n" in response:
        return response.split("\n")[0].strip()
    else:
        return response.strip() or "No response."

# ---------- MULTI-MODAL CHAT FUNCTION ----------
def chat_with_model(user_text, image_path=None):
    """
    Combine user text and optional image caption for multi-modal response
    and return exactly one concise sentence
    """
    if image_path:
        caption = generate_caption(image_path)
        prompt = f"Describe this image in ONE concise sentence: {caption}"
    else:
        prompt = f"Answer in ONE concise sentence: {user_text}"

    return generate_response(prompt, max_tokens=60)

