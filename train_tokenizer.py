import random
import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

random.seed(42)

def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<unk>", "<s>", "</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    data_path = 'dataset/pretrain_hq.jsonl'

    texts = read_texts_from_jsonl(data_path)

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.decoder = decoders.ByteLevel()

    tokenizer_dir = "model/yuanmeng_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<unk>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 YuanMeng，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)
    
    print("Tokenizer training completed and saved")

def eval_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("model/yuanmeng_tokenizer")

    messages = [
        {"role": "system", "content": "你是缘梦大模型, 最有用的小助手！"},
        {"role": "user", "content": "你叫什么名字？"},
        {"role": "assistant", "content": "我叫缘梦, 很高兴能和你聊天"}
    ]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    print(new_prompt)

    actual_vocab_size = len(tokenizer)
    print('Actual vocabulary size of the tokenizer:', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('Length of the encoder:', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("Is the decoded text identical to the original input:", response == new_prompt)

if __name__ == '__main__':
    train_tokenizer()
    eval_tokenizer()