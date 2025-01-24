"""
Date: 2024-01-24
Description: The code is improved and integrated from "LLMs-from-scratch".
If there is anything unclear, please refer to the original book.
"""


import subprocess


def test_gpt_class_finetune():
    command = ["python", "gpt_instruction_finetuning.py", "--test_mode"]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"

if __name__ == '__main__':
    test_gpt_class_finetune()