YOUR_PATH = "YOUR_MODEL_PATH"

MODELS = {
    'llama2': {
        'model_path': YOUR_PATH+'/Llama-2-7b-chat-hf',
        'tokenizer_path': YOUR_PATH+'/Llama-2-7b-chat-hf',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_path': YOUR_PATH+'/vicuna-13b-v1.5',
        'tokenizer_path': YOUR_PATH+'/vicuna-13b-v1.5',
        'conversation_template': 'vicuna'
    },
    'vicuna7b': {
        'model_path': YOUR_PATH+'/vicuna-7b-v1.5',
        'tokenizer_path': YOUR_PATH+'/vicuna-7b-v1.5',
        'conversation_template': 'vicuna_v1.1'
    },
    'opt': {
        'model_path': YOUR_PATH+'/opt-1.3b',
        'tokenizer_path': YOUR_PATH+'/opt-1.3b',
        'conversation_template': 'opt'
    },
    'mistral': {
        'model_path': YOUR_PATH+'/Mistral-7B-v0.1',
        'tokenizer_path': YOUR_PATH+'/Mistral-7B-v0.1',
        'conversation_template': 'mistral'
    },
    'chatglm3': {
        'model_path': YOUR_PATH+'/chatglm3-6b',
        'tokenizer_path': YOUR_PATH+'/chatglm3-6b',
        'conversation_template': 'chatglm3'
    },
    'qwen': {
        'model_path': YOUR_PATH+'/Qwen-7B-Chat',
        'tokenizer_path': YOUR_PATH+'/Qwen-7B-Chat',
        'conversation_template': 'qwen-7b-chat'
    },
    'gpt2': {
        'model_path': YOUR_PATH+'/gpt2',
        'tokenizer_path': YOUR_PATH+'/gpt2',
        'conversation_template': 'gpt2'
    },
    'tinyllama': {
        'model_path': YOUR_PATH+'/TinyLlama-1.1B-Chat-v1.0',
        'tokenizer_path': YOUR_PATH+'/TinyLlama-1.1B-Chat-v1.0',
        'conversation_template': 'llama-2'
    },
    'tinyvicuna': {
        'model_path': YOUR_PATH+'/Tiny-Vicuna-1B',
        'tokenizer_path': YOUR_PATH+'/Tiny-Vicuna-1B',
        'conversation_template': 'vicuna'
    }
}

default_args = {
    'batch_size': 4,
    'lr': 2e-5,
    'epoch': 3,
    'accumulation_steps': 4,
    'qa_path': "./data/TruthfulQA.csv",
    'static': False,
    'norm_embedding': True,
    'use_lora': False,  
    'save_every': 1,  # "How many epoches to save model."
    'model_save_dir': "./models/",   #repeat_models
    "note":"Please ensure that your response is harmless or refused.\n",
    #extra args
    "full_finetuning": False,
    "log": False,
    "lamda": 1,   #connection    1
    "alpha": 0.5, #mask           0.5
    "beta": 0., #ppl            0.01
    "r": 0.5,   # 0.5
    "max_new_tokens": 400,
    "space_token_id": "", #perturbation    <unk>/random/space/./!
}
