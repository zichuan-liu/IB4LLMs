import os
import torch
import numpy as np
import pandas as pd
import random as rd

from tqdm.auto import tqdm
import argparse
import json
import time

import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs

rd.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def get_method(args, target_model): 
    if args.method == "smooth": 
        # Create SmoothLLM instance
        defense = defenses.SmoothLLM(
            target_model=target_model,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies,
            attack_type = args.attack
        )
    elif args.method == "ra": 
        # Create RALLM instance
        defense = defenses.RALLM(
            target_model=target_model,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies,
            attack_type = args.attack
        )
    elif args.method == "selfdefense":
        # Create SelfDefenseLLM instance
        defense = defenses.SelfDefenseLLM(
            target_model=target_model,
            attack_type = args.attack
        )
    elif args.method == "sft":
        # Create SFT instance
        defense = defenses.Finetuning(
            target_model=target_model,
            attack_type = args.attack
        )
        is_train="./models/"
        if not os.path.exists(is_train+"{}_{}/".format(args.method, target_model.model_name)):
            defense.train()
        else:
            defense.load_pretrain_model(device="cuda:"+args.cuda)

    elif args.method == "unlearning":
        # Create unlearning instance
        defense = defenses.Unlearning(
            target_model=target_model,
            attack_type = args.attack
        )
        is_train="./models/"
        if not os.path.exists(is_train+"{}_{}/".format(args.method, target_model.model_name)):
            defense.train()
        else:
            defense.load_pretrain_model(device="cuda:"+args.cuda)
    elif args.method == "semantic":
        # Create semanticsmooth instance
        # config = model_configs.MODELS[args.protect_model]
        # protect_model = language_models.LLM(
        #     model_name=args.protect_model,
        #     model_path=config['model_path'],
        #     tokenizer_path=config['tokenizer_path'],
        #     conv_template_name=config['conversation_template'],
        #     device="cuda:"+args.cuda,
        #     is_small=True
        # )
        defense = defenses.SemanticSmooth(
            protect_model=target_model, # same to authors papers
            target_model=target_model,
            attack_type = args.attack
        )
    elif args.method == "vib":
        # Create ours instance
        config = model_configs.MODELS[args.protect_model]
        protect_model = language_models.LLM(
            model_name=args.protect_model,
            model_path=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            conv_template_name=config['conversation_template'],
            device="cuda:"+args.cuda,
            is_small=True
        )
        defense = defenses.VIBLLM(
            protect_model=protect_model,
            target_model=target_model,
            attack_type = args.attack,
            space_token_id =args.space_token_id
        )
        model_configs.default_args["space_token_id"] = args.space_token_id if args.space_token_id!="" else model_configs.default_args["space_token_id"]
    elif args.method == "none":
        # Create Original instance
        defense = defenses.Defense( # TODO:fix
            target_model=target_model,
            attack_type = args.attack
        )
    else:
        # Create Original instance
        print("WARNING: This Protection Method Is NOT Implemented. Setting NO Defense By Default.\n"*3)
        defense = defenses.Defense( # TODO:fix
            target_model=target_model
        )
    return defense


def main(args):

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    print("args", args)

    # Instantiate the targeted LLM
    if args.target_model=="chatgpt" or args.target_model=="gpt4":
            target_model = language_models.GPT(
            model_name=args.target_model,
            device="cuda:"+args.cuda
    )
    else:
        config = model_configs.MODELS[args.target_model]
        target_model = language_models.LLM(
            model_name=args.target_model,
            model_path=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            conv_template_name=config['conversation_template'],
            device="cuda:"+args.cuda
        )
    # Create attack instance, used to create prompts
    if args.multi:
        attack = vars(attacks)[args.attack](
            target_model=target_model,
            attack=["PAIR"],
            logfile=args.attack_logfile,
            multi = args.multi
        )
    else:
        attack = vars(attacks)[args.attack](
            logfile=args.attack_logfile,
            target_model=target_model
        )
    defense = get_method(args, target_model)

    base_path = args.attack+"_"+args.method+"_"+target_model.model_name

    if args.method == "vib":
        print("Using perturbation:", model_configs.default_args["space_token_id"], model_configs.default_args["space_token_id"])
        base_path = args.attack+"_"+args.method+"_"+target_model.model_name+"_"+model_configs.default_args["space_token_id"]
    if args.multi:
        base_path ="Trans_"+base_path
        if args.smoothllm_num_copies!=1 and args.method == "smooth":
            base_path += "_" + str(args.smoothllm_num_copies)
    elif args.ab_mode:
        string = "full" if args.fullft else "mlp"
        string = string + "_" + str(args.alpha)+ "_"+ str(args.lamda)+ "_"+ str(args.r)
        base_path = base_path+"_"+string+"_"+str(args.p)

    if not os.path.exists(args.results_dir+"/"+base_path):
        os.makedirs(args.results_dir+"/"+base_path)
    output_file = "result.jsonl"
    
    jailbroken_results = []
    runtimes = []

    if os.path.exists(args.results_dir+"/"+base_path+"/"+output_file):
        os.remove(args.results_dir+"/"+base_path+"/"+output_file)

    # load small model
    if args.method == "vib":
        if args.ab_mode:
            defense.load_pretrain_model(f"./ablation_models/vib_tinyllama_{string}_vicuna_/")
        elif not args.multi:
            defense.load_pretrain_model()
        else:
            defense.load_pretrain_model("./models/vib_tinyllama_vicuna_/")
            
    with open(os.path.join(args.results_dir+"/"+base_path, output_file), "a") as f:
        for i, prompt in tqdm(enumerate(attack.prompts)):
            if i>=args.test_num and (args.attack=="PAIR" or args.attack=="GCG"):# 
                break
            print("="*20,i,"="*20)
            time0 = time.time()
            if args.method == "vib":
                if args.ab_mode:
                    output, sub_x = defense(prompt, top_p=args.p)
                else:
                    output, sub_x = defense(prompt)
            elif args.method == "smooth" or args.method == "ra" or args.method == "semantic":
                output, sub_x = defense(prompt)
            else:
                output = defense(prompt)
            jb = defense.is_jailbroken(output)
            jailbroken_results.append(jb)
            # print(prompt.full_prompt)
            time1 = time.time()
            eta = (time1 - time0)
            runtimes.append(eta)
            eta_deal = time.strftime("%Hh%Mm%Ss", time.gmtime(eta))
            
            pr_sys, pr, def_result = prompt.full_prompt, prompt.perturbable_prompt, output
            print("Original Attack:", pr )
            print("-------------------------------------")
            print(jb, args.method, "defense output:", output)

            if args.method == "vib" or args.method == "smooth" or args.method == "ra" or args.method == "semantic":
                f.write(json.dumps(
                    {"prompt_with_sys": pr_sys, "prompt": pr, "sub_prompt": sub_x, "def_result": def_result, "eta": eta, "eta_deal": eta_deal,"jb":jb}
                    ) + "\n")
                f.flush()
            else:
                f.write(json.dumps(
                    {"prompt_with_sys": pr_sys, "prompt": pr, "def_result": def_result, "eta": eta, "eta_deal": eta_deal,"jb":jb}
                    ) + "\n")
                f.flush()
            if torch.cuda.is_available():  
                torch.cuda.empty_cache()  

    print('{} made errors'.format(args.method), np.mean(jailbroken_results) * 100)

    # Save results to a pandas DataFrame
    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [args.smoothllm_num_copies],
        'Runtime': "{:.3f}$\pm${:.3f}".format(np.mean(runtimes), np.std(runtimes)),
        'Perturbation percentage': [args.smoothllm_pert_pct],
        'len': [len(jailbroken_results)],
        'JB percentage': [np.mean(jailbroken_results) * 100],
        'Trial index': [args.trial]
    })
    summary_df.to_csv(os.path.join(
        args.results_dir+"/"+base_path, 'summary.csv'
    ), index=False)
    print(summary_df.head())
    print("Finished!")


# python main.py  --results_dir ./our_results  --target_model vicuna  --attack GCG
# python main.py  --results_dir ./transfer_results  --target_model llama2  --attack EasyJailbreak
if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda',
        type=str,
        default='0'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./our_results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )
    parser.add_argument(
        '--test_num',
        type=int,
        default=120
    )
    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2', 'vicuna7b', 'mistral', 'chatglm3', 'qwen', 'chatgpt', 'gpt4']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='PAIR',
        choices=['GCG', 'PAIR', 'EasyJailbreak', 'TriviaQA']
    )

    # method
    parser.add_argument(
        '--method',
        type=str,
        default='none'
    )

    # transfor
    parser.add_argument(
        '--multi',
        type=bool,
        default=False
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',   # for fairness
        type=int,
        default=1,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomPatchPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )

    # Ours Protect LLM
    parser.add_argument(
        '--protect_model',
        type=str,
        default='tinyllama',
        choices=['llama2', "tinyllama", "vicuna7b", "tinyvicuna"]
    )
    parser.add_argument(
        '--ab_mode',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--lamda',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '--r',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--fullft',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--p',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--space_token_id',
        type=str,
        default=""
    )

    args = parser.parse_args()

    args.attack_logfile = ""
    if args.target_model=='vicuna':
        if args.attack=="PAIR":
            args.attack_logfile = "./data/jailbreaking_vicuna.csv"
        elif args.attack=='GCG':
            args.attack_logfile = "./data/GCG_new/individual_behavior_controls_vicuna.json"
        else:
            args.attack_logfile = "./data/jailbreaking_vicuna.csv"
    elif args.target_model=='llama2':
        if args.attack=="PAIR":
            args.attack_logfile = "./data/jailbreaking_llama-2.csv"
        elif args.attack=='GCG':
            args.attack_logfile = "./data/GCG_new/individual_behavior_controls_llama2.json"
        else:
            args.attack_logfile = "./data/jailbreaking_llama-2.csv"
    else:
        pass
    main(args)