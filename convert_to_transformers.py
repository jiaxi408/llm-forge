import os
import torch
from transformers import AutoTokenizer

CKPT_PATH = "out/sft_512.pth"
TOKENIZER_DIR = "model/yuanmeng_tokenizer"
OUTPUT_DIR = "Yuanmeng_Models/yuanmeng-26m-instruct"

from model.config import YuanMengConfig
from model.model import YuanMengModel

if os.path.isdir(OUTPUT_DIR):
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

sd = torch.load(CKPT_PATH, map_location='cpu')

config = YuanMengConfig()
model = YuanMengModel(config)
model.load_state_dict(sd)

model.eval()

# Safetensors must be disabled; otherwise, shared weight verification will fail
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)

AutoTokenizer.from_pretrained(TOKENIZER_DIR).save_pretrained(OUTPUT_DIR)