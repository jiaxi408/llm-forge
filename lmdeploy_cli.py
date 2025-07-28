from transformers import AutoConfig, AutoModelForCausalLM
from model.config import YuanMengConfig
from model.model import YuanMengModel

AutoConfig.register("yuanmeng", YuanMengConfig)
AutoModelForCausalLM.register(YuanMengConfig, YuanMengModel)

from lmdeploy.cli.entrypoint import run

if __name__ == "__main__":
    run()