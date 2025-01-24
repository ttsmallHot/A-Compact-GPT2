# A-Compact-GPT2

- This project write a GPT2 from scratch, pretrain and fine tune it to enable dialogue.
- all of the Large Language Model code can be run on CPU
- The code is mainly improved and integrated from "LLMs-from-scratch". If there is anything unclear, please refer to the original book.

## GPT2 architecture

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/13.webp?1" width="400px">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/15.webp" width="400px">

## pretrain

- please run the following shell for pretrain

```shell
python gpt_pretrain.py
```

## fine-tune

- For better final results, we use gpt2 to pretrain weights and then fine tune them instead of the weights we previously trained
- please run the following shell for test fine-tune

```
python gpt_instruction_finetuning.py --test_mode
```

- please run the following shell for fine-tune

```
python gpt_instruction_finetuning.py
```

## chat

- you can chat with your fine tuned model by following shell

```
python play.py
```

