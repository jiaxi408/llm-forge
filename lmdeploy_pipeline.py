from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from transformers import AutoConfig, AutoModelForCausalLM
from chat_templates import get_minimind_template

model = "Yuanmeng_Models/yuanmeng-26m-instruct"
# model = "Yuanmeng_Models/yuanmeng-26m-instruct-w4a16-4bit"
# model = "Shanghai_AI_Laboratory/internlm3-8b-instruct-awq"

if model in ["Yuanmeng_Models/yuanmeng-26m-instruct", "Yuanmeng_Models/yuanmeng-26m-instruct-w4a16-4bit"]:
    from lmdeploy.pytorch.models.yuanmeng import YuanMengConfig, YuanMengModel
    AutoConfig.register("yuanmeng", YuanMengConfig)
    AutoModelForCausalLM.register(YuanMengConfig, YuanMengModel)
    tmpl_cfg = get_minimind_template()
else:
    tmpl_cfg = None

backend_cfg = TurbomindEngineConfig(session_len=2048*32, quant_policy=4)

gen_config = GenerationConfig(max_new_tokens=2048, top_p=0.8, top_k=40, temperature=0.8, do_sample=True)

history = []

with pipeline(
    model,
    backend_config=backend_cfg,
    chat_template_config=tmpl_cfg
) as pipe:
    while True:
        user_input = input("🤔: ")
        if user_input.strip().lower() == "exit":
            print("🤖: 拜拜ヾ(•ω•`)o")
            break

        history.append({"role": "user", "content": user_input})

        print("🤖: ", end="", flush=True)

        answer_text = ""
        for item in pipe.stream_infer(history, gen_config=gen_config):
            if item.text.strip():
                print(item.text, end="", flush=True)
                answer_text += item.text

        print()

        history.append({"role": "assistant", "content": answer_text})