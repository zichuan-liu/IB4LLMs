import json
import pandas as pd
import lib.defenses as defenses
from datasets import load_dataset  
import random

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model


class GCG(Attack):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        self.goals = log['goal']
        self.targets = log['target']
        self.controls = log['controls']

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]

    def create_prompt(self, goal, control, target, max_new_len=150):
        """Create GCG prompt."""

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2,
            max_new_len
        )

        # Create full prompt for LLM
        conv_template = self.target_model.conv_template
        conv_template.append_message(
            conv_template.roles[0], f"{goal} {control}"
        )
        conv_template.append_message(conv_template.roles[1], f"")
        prompt = conv_template.get_prompt()

        # As per the GCG source code, we encode then decode the full prompt
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode(
            encoding.input_ids
        ).replace('<s>','').replace('</s>','')

        # Clear the conv template
        conv_template.messages = []
        
        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        return Prompt(
            full_prompt, 
            perturbable_prompt, 
            max_new_tokens
        )

class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)

        df = pd.read_csv(logfile)
        jailbreak_prompts = df['attack_prompt'].to_list()
        
        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        
    def create_prompt(self, prompt):        

        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None) # TODO:gpt-4/palm 
        full_prompt = conv_template.get_prompt()
        
        # Clear the conv template
        conv_template.messages = []
        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=150
        )


class EasyJailbreak(Attack):

    """EasyJailbreak.
    EasyJailbreak is an easy-to-use Python framework designed for researchers and developers focusing on LLM security. 
    Specifically, EasyJailbreak decomposes the mainstream jailbreaking process into several iterable steps: 
    initialize mutation seeds, select suitable seeds, add constraint, mutate, attack, and evaluate. On this basis, 
    EasyJailbreak provides a component for each step, constructing a playground for further research and attempts.
    Paper: https://github.com/EasyJailbreak
    """

    def __init__(self, target_model, attack=["Autodan", "GPTFuzz", "ReNellm"], logfile="", multi=False):
        super(EasyJailbreak, self).__init__(logfile, target_model)
        jaibreak_methods = attack
        if target_model.model_name=="llama2":
            llm = "llama27b"
        elif target_model.model_name=="vicuna":
            llm = "vicuna13b"
        elif target_model.model_name=="vicuna7b":
            llm = "vicuna7b"
        elif target_model.model_name=="mistral":
            llm = "mistral7b"
        elif target_model.model_name=="qwen":
            llm = "qwen7b"
        elif target_model.model_name=="chatglm3":
            llm = "chatglm3"
        else:
            llm = target_model.model_name

        self.target_model_name= target_model.model_name

        defense = defenses.Defense(target_model)
        self._test_prefixes = defense.TEST_PREFIXES

        jailbreak_prompts = []
        for method in jaibreak_methods:
            if method=="PAIR":
                if target_model.model_name=="llama2" or target_model.model_name=="vicuna":
                    df = pd.read_csv(logfile)
                    jailbreak_prompts = df['attack_prompt'].to_list()
                    jailbreak_prompts = jailbreak_prompts[:120]
                    break

            file_path = f"./data/EasyJaibreak_results/{method}/{llm}.jsonl"
            data = []
            with open(file_path) as f:
                for line in f:
                    data.append(json.loads(line))

            jailbreak_prompt = []
            att_target_responses = []
            his_ref = []

            for d in data:
                if "[PROMPT]" in d['jailbreak_prompt']:
                    jp = d['jailbreak_prompt'].replace("[PROMPT]", d['query'])
                    jp = jp.replace("{query}", "")
                else:
                    jp = d['jailbreak_prompt'].replace("{query}", d['query'])
                
                res = d['target_responses'][0]
                if not multi:
                    if self.not_matched(res):
                        att_target_responses.append(res)
                        his_ref.append(d['eval_results'])
                        jailbreak_prompt.append(jp)
                else:
                    att_target_responses.append(res)
                    his_ref.append(d['eval_results'])
                    jailbreak_prompt.append(jp)
            jailbreak_prompts += jailbreak_prompt

        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        print("=========================================================")
        print("The number of EZjailbreaking prompts is:", len(self.prompts))
        print("=========================================================")

    def create_prompt(self, prompt):        

        conv_template = self.target_model.conv_template
        if self.target_model_name=="llama2":    # because the ezjailbreak is fastchat=0.2.23
            conv_template.system='[INST] ' # here need to update https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None) # TODO:gpt-4/palm 
        full_prompt = conv_template.get_prompt()
        
        # Clear the conv template
        conv_template.messages = []
        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=250
        )

    def not_matched(self, o):
        for t in self._test_prefixes:
            if t in o:
                return False
        return True




class TriviaQA(Attack):

    """
    TriviaqQA is a reading comprehension dataset containing over 650K question-answer-evidence triples. 
    TriviaqQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents,
      six per question on average, that provide high quality distant supervision for answering the questions.
    URL: https://huggingface.co/datasets/mandarjoshi/trivia_qa
    """

    def __init__(self, logfile, target_model):
        super(TriviaQA, self).__init__(logfile, target_model)

        unfiltered_dataset = load_dataset('trivia_qa', 'unfiltered.nocontext', split='train[:50%]')
        sample_num=230
        question = [q for q in unfiltered_dataset['question'] if len(q)>sample_num]
        sampled_data = random.sample(list(question), sample_num)  
        self.prompts = [
            self.create_prompt(prompt)
            for prompt in sampled_data
        ]
        
    def create_prompt(self, prompt):        

        conv_template = self.target_model.conv_template
        # We use Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM
        # Default system prompt has unnecessary negative effect on benign answer rate!
        conv_template.system='' # here need to update https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None) # TODO:gpt-4/palm 
        full_prompt = conv_template.get_prompt()
        
        # Clear the conv template
        conv_template.messages = []
        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=150
        )
