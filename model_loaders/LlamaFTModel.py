from .BaseModel import BaseModel
from .LlamaInstructModel import LlamaInstructModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
#import torch 

class LlamaFTModel(LlamaInstructModel):
    def load(self, model_args):
        print(self.model_id)
        model_name = self.model_id
        model_dir = model_args['model_dir']

        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",        # same as your original dtype
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer, self.llm


    def prompt_response(self, system_prompt, user_prompt):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}: {user_prompt}"}],
            tokenize=False,
            add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None,
        )

        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        #if "assistant" in response_text:
        #    response_text = response_text.split("assistant", 1)[1].strip()

        return response_text
