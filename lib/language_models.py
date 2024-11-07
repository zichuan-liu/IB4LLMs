import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from transformers import BitsAndBytesConfig
from typing import Dict, List
import openai
import re  
import os
import time

AZURE = True
YOUR_URL = "YOUR_URL"
YOUR_KEY = "YOUR_KEY"
if not AZURE:
    from openai import OpenAI
    NOT_AZURE_KEY = YOUR_KEY
    NOT_AZURE_URL = YOUR_URL
else:
    AZURE_OPENAI_ENDPOINT = YOUR_URL
    AZURE_API_VERSION = "2024-02-15-preview"
    AZURE_OPENAI_KEY = YOUR_KEY
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
    os.environ["AZURE_OPENAI_KEY"] = AZURE_OPENAI_KEY

class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model_name,
        model_path, 
        tokenizer_path, 
        conv_template_name,
        device,
        is_small = False,
        is_bnb = False,
    ):
        self.model_name = model_name
        self.conv_template_name = conv_template_name
        self.model_path = model_path

        self.device = device
        # Language model
        if is_bnb:
            nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            ).eval()
        elif is_small:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            ).to(device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                ).eval()
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, max_new_tokens=150, 
                        temperature=0, top_p: float = 1.0,):
        inputs = self.tokenizer(batch, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        batch_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        # # Pass current batch through the tokenizer
        # batch_inputs = self.tokenizer(
        #     batch, 
        #     padding=True, 
        #     truncation=False, 
        #     return_tensors='pt'
        # )
        # batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        # batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # # Forward pass through the LLM
        # try:
        #     outputs = self.model.generate(
        #         batch_input_ids, 
        #         attention_mask=batch_attention_mask, 
        #         max_new_tokens=max_new_tokens,
        #         )
        # except RuntimeError:
        #     return []

        # # Decode the outputs produced by the LLM
        # batch_outputs = self.tokenizer.batch_decode(
        #     outputs, 
        #     skip_special_tokens=True
        # )
        # gen_start_idx = [
        #     len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
        #     for i in range(len(batch_input_ids))
        # ]
        # batch_outputs = [
        #     output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        # ]

        return batch_outputs


class GPTLanguageModel():
    # same as EasyJailbreak's version
    # https://github.com/search?q=repo%3AEasyJailbreak%2FEasyJailbreak+OpenaiModel&type=code&p=2
    def __init__(self, model_name="gpt-35-turbo",device="cuda:0"):
        if model_name=="chatgpt":
            if AZURE:
                self.model_eng_name = "gpt-35-turbo"
            else:
                self.model_eng_name = "gpt-3.5-turbo-0301"
        elif model_name=="gpt4":
            if AZURE:
                self.model_eng_name = "gpt-4"
            else:
                self.model_eng_name = "gpt-4-1106-preview"
        else:
            self.model_eng_name = model_name
        
        self.model_name = model_name        #   gpt-35-turbo  gpt-4
        self.conv_template = get_conversation_template('chatgpt')

        self.device = device


    def __call__(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class GPT(GPTLanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 10
    API_TIMEOUT = 20
    if AZURE:
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
        openai.api_version = AZURE_API_VERSION

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        prompt = re.findall(r'### Human:\s*(.*?)\s*### Assistant:', conv, re.DOTALL)[-1].strip()
        conv_template = get_conversation_template(
            self.model_name
        )
        # print("---")
        # print(prompt)
        # print("---")
        conv_template.append_message(conv_template.roles[0], prompt)

        output = self.API_ERROR_OUTPUT
        if AZURE:
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = openai.ChatCompletion.create(
                                engine = self.model_eng_name,
                                messages = conv_template.to_openai_api_messages(),
                                max_tokens = max_n_tokens,
                                temperature = temperature,
                                top_p = top_p,
                                request_timeout = self.API_TIMEOUT,
                                )
                    output = response["choices"][0]["message"]["content"]
                    break
                except openai.error.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)
            
                time.sleep(self.API_QUERY_SLEEP)
        else:
            output = self.openaichat(my_conv=conv_template.to_openai_api_messages())
        return output 
    
    def __call__(self, 
                        batch,
                        max_new_tokens: int, 
                        temperature: float = 0.,
                        top_p: float = 1.0,):
        if len(batch)>10:   # str
            return [self.generate(conv, max_new_tokens, temperature, top_p) for conv in [batch]]
        else:   # [str]
            return [self.generate(conv, max_new_tokens, temperature, top_p) for conv in batch]


    def openaichat(self, my_conv=None, system=None,user=None,assistant=None,top_p=1e-8, temperature=1, max_tokens=1024, **kwargs):
        client = OpenAI(
            api_key=NOT_AZURE_KEY,
            base_url=NOT_AZURE_URL
        )
        if my_conv is not None:
            messages = my_conv
        else:
            messages=[]
            if system is not None:
                messages.append({'role':'system','content':system})
            if user is not None:
                messages.append({'role': 'user', 'content': user})
            if assistant is not None:
                messages.append({'role': 'assistant', 'content': assistant})
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = client.chat.completions.create(
                    # model="gpt-4-1106-preview",
                    # model="gpt-3.5-turbo-1106",
                    # model="gpt-4-0613",
                    model=self.model_eng_name,
                    messages=messages,
                    top_p=top_p,temperature=temperature,max_tokens=max_tokens,**kwargs
                )
                completion_text=completion.choices[0].message.content
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return completion_text