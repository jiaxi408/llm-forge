import torch
from model.config import YuanMengConfig
from model.model import YuanMengModel

# Arg set
# mode = 0 indicates the chat mode of a pre-trained model
# mode = 1 indicates the chat mode of an SFT model
mode = 1
max_seq_len = 256
max_new_tokens = 64
temperature = 0.75
top_p = 0.90

config = YuanMengConfig(max_seq_len=max_seq_len)
model = YuanMengModel(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if mode == 0:
    state_dict = torch.load('out/pretrain_512.pth', map_location=device)

elif mode == 1:
    state_dict = torch.load('out/sft_512.pth', map_location=device)

model.load_state_dict(state_dict)

with torch.no_grad():
    messages = []
    while True:
        user_input = input("🤔: ")
        if user_input.strip().lower() == 'exit':
            print("🤖: 拜拜ヾ(•ω•`)o")
            break
        
        if mode == 0:
            messages = user_input
        elif mode == 1:
            messages.append({"role": "user", "content": user_input})
        print("🤖: ", end="", flush=True)

        answer_text = ""
        for item in model.chat(messages=messages, mode=mode, stream=True, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p):
            if "errors" in item:
                break
            print(item["text"], end="", flush=True)
            answer_text += item["text"]
        
        print()

        if mode == 1:
            messages.append({"role": "assistant", "content": answer_text})