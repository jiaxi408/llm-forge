from lmdeploy import ChatTemplateConfig

def get_minimind_template():
    return ChatTemplateConfig(
        model_name="yuanmeng",
        system="<s>system\n",
        meta_instruction='你是 YuanMeng，是一个有用的人工智能助手。',
        eosys="</s>\n",
        user="<s>user\n",
        eoh="</s>\n",
        assistant="<s>assistant\n",
        eoa="</s>",
        separator="\n",
        capability="chat",
    )