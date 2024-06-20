import time

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)
from utils.prompter import Prompter

prompt_template = ""
base_model = "meta-llama/Meta-Llama-3-8B"
prompter = Prompter(prompt_template)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    # quantization_config=quantization_config,
)
lora_weights = "/home/arg/llama-peft-bt/output-test-3/checkpoint-60"
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
    # quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()
instruction = """
You will be provided a summary of a task performed by a behavior tree, and your objective is to express this behavior tree in symbolic format.

-> means sequence logic
? means fallback logic
|| N means parallel logic executing N nodes
"""
input_text = "The behavior tree describes a task where an object is first checked to be in view. Once the object is confirmed to be in view, two parallel controls are executed: one controlling the linear movement along the x-axis and the other controlling the linear movement along the y-axis."
input_text = "The behavior tree first checks if there is any object in view. If there is an object, it simultaneously executes control actions for linear x and linear y directions. If no object is in view, the robot will then proceed to explore a pattern block."
prompt = prompter.generate_prompt(instruction, input_text)
inputs = tokenizer(prompt, return_tensors="pt")
device = "cuda"
input_ids = inputs["input_ids"].to(device)
# temperature = 0.1
# top_p = 0.75
# top_k = 40
# num_beams = 4
max_new_tokens = 128
generation_config = GenerationConfig(
    do_sample=False,
    num_beams=1,
)
for _ in range(10):
    start = time.time()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    print(f"Time taken: {time.time() - start}")
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Output:\n\n")
    print(prompter.get_response(output).split("<|end_of_text|>")[0])
