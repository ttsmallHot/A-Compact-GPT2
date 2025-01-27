import torch
from colorama import Style, Fore
from utils import (
    calc_loss_loader,
    generate,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)
from gpt_architecture import GPTModel
import tiktoken

RESULT_PREFIX = "result/"

def format_play_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry}"
    )

    return instruction_text

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    BASE_CONFIG = {
        "vocab_size": 50257,  # 字典大小
        "context_length": 1024,  # 长度
        "drop_rate": 0.0,  # Dropout 比例
        "qkv_bias": True  # Query-key-value 偏差
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model = GPTModel(BASE_CONFIG)

    # 这里应该改成你要使用的模型
    model.load_state_dict(torch.load(RESULT_PREFIX + "gpt2-medium355M-sft-standalone.pth", weights_only=True))
    model.eval()

    print("Loaded model:", CHOOSE_MODEL)

    caption = '|{}{}{}|'.format(' ' * 3, "chat begins, press 'q' to quit the conversation", ' ' * 3)
    print("-" * len(caption))
    print(caption)
    print("-" * len(caption))


    # 进入主循环
    while True:
        # 是一个格式化字符串（f-string），用于显示提示信息，类似于format或者直接逗号
        # {Fore.YELLOW}：这是 colorama.Fore 模块中的一个常量，用于将后续文本的颜色设置为黄色
        # {Style.RESET_ALL}：这是 colorama.Style 模块中的一个常量，用于重置所有样式（包括颜色、背景色、文字效果等），让后续的输出恢复默认样式。
        #.lower() 是字符串的方法，将输入的文本转换为小写字母。即使用户输入的是大写字母，它也会被转换成小写字母。
        user_dialog = input(f" {Fore.YELLOW}User:{Style.RESET_ALL} ").lower()
        if user_dialog == "q" or user_dialog == "quit":
            break

        input_text = format_play_input(user_dialog)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        print(Fore.BLUE, "chatbot:", Style.RESET_ALL, response_text)