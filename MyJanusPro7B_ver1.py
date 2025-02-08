#Install Janus Pro 7B
#https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
#https://www.youtube.com/watch?v=ZIjrq3Rzn1o
#https://www.danielcorin.com/til/deekseek/janus-pro-local/
#https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus
#pip install transformers
#pip install janus
#pip install gradio==3.50
#pip install gradio
#pip3 install torch torchvision torchaudio
#Error encountered is <ModuleNotFoundError: No module named 'janus.models'> because MyJanusPro7B is not inside Janus folder.
#git clone https://github.com/deepseek-ai/Janus.git


#git clone https://github.com/deepseek-ai/Janus.git
import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()


#New 1 = Prompt User to Describe Image
description = input("Please describe the image you want to generate: ")

conversation = [
    {
        "role": "<|User|>",
        "content": description,
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
        device
    )
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).to(device)

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs("generated_samples", exist_ok=True)
    
    # New 2 = Find next available finename
    existing_files = os.listdir("generated_samples")
    existing_indices =[
        int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("img_")
    ]
    next_index = max(existing_indices, default=-1) + 1
    
    for i in range(parallel_size):
        save_path = os.path.join("generated_samples", "img_{}.jpg".format(next_index + i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
    parallel_size=1, # to generate a single image
)
