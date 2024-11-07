import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
import json

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_truthfulqa_dataloader(tokenizer, tqa_file_path="data/TruthfulQA.csv", batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv(tqa_file_path)
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)

    # Split train/val/test = 0.8/0.2.
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    # print("create train_qa_data len:", train_len)
    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, test_dataloader



def create_attack_dataloader(tokenizer, target_model_name="vicuna", batch_size=4, max_new_tokens=400):
    """
    Create the fintuning dataloader for the attack data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of advbench normal Q&A pairs.
    """
    if target_model_name=="vicuna":
        file_path=f"./data/jailbreaking_vicuna.csv"
    elif target_model_name=="llama2":
        file_path=f"./data/jailbreaking_llama-2.csv"
    else:
        print("The dataset does not exist.")
        exit()

    df = pd.read_csv(file_path)
    df = df.iloc[120:]  # the frist 120 is the test data
    questions = df['attack_prompt'].to_list()
    good_answers = df["normal_response"].to_list()

    # it can be added, but it is unstable, the response are all in /n/n/n
    if target_model_name=="vicuna":
        file_path=f"./data/GCG_new/individual_behavior_controls_vicuna.json"
    elif target_model_name=="llama2":
        file_path=f"./data/GCG_new/individual_behavior_controls_llama2.json"
    else:
        print("The dataset does not exist.")
        exit()

    with open(file_path, 'r') as f:
        log = json.load(f)
    questions2 = [f"{goal} {control}" for goal, control in zip(log['goal'],log['controls'])][120:]
    controls = df["normal_response"].to_list()
    questions = questions+questions2
    good_answers = good_answers+controls

    sampled_data = load_dataset('trivia_qa', 'unfiltered.nocontext', split='train[:400]')
    questions = questions+[item['question'] for item in sampled_data]
    good_answers = good_answers+[item['answer']['value'] for item in sampled_data]
    
    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {good_answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader


def create_ib_attack_dataloader(tokenizer, target_model_name="vicuna", batch_size=4, max_new_tokens=400, test_data=False):
    """
    Create the advbench dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of advbench normal Q&A pairs.
    """
    data = {"input_ids": [], "attention_mask": [], "q_input_id_lens": [], "prompt_input_id_lens": []}
    # target_model_name="llama2" # olny load llama2, because vicuna data is unclear and have many error that is not im sorry

    if target_model_name=="vicuna":
        file_path=f"./data/jailbreaking_vicuna.csv"
    elif target_model_name=="llama2":
        file_path=f"./data/jailbreaking_llama-2.csv"
    else:
        print("The dataset does not exist.")
        exit()

    df = pd.read_csv(file_path)
    df = df.iloc[120:]  # the frist 120 is the test data
    questions, ori_questions = df['attack_prompt'].to_list(), df["orig_goal"].to_list()
    good_answers = df["normal_response"].to_list()

    if target_model_name=="vicuna":
        file_path=f"./data/GCG_new/individual_behavior_controls_vicuna.json"
    elif target_model_name=="llama2":
        file_path=f"./data/GCG_new/individual_behavior_controls_llama2.json"
    else:
        print("The dataset does not exist.")
        exit()

    with open(file_path, 'r') as f:
        log = json.load(f)
    questions2 = [f"{goal} {control}" for goal, control in zip(log['goal'],log['controls'])][120:]
    controls = df["normal_response"].to_list()
    questions = questions+questions2
    good_answers = good_answers+controls

    sampled_data = load_dataset('trivia_qa', 'unfiltered.nocontext', split='train[:400]')
    questions = questions+[item['question'] for item in sampled_data]
    good_answers = good_answers+[item['answer']['value'] for item in sampled_data]

    print("ALL DATA LENGTH IS", len(good_answers))

    for question, good_answer in zip(questions, good_answers):
        # compression
        # text = f"### Summarize the follow sentence into a brief sentence, and end with '?' if the sentence is a question: {question}\n ### Summarization: {ori_question}"
        text = f"### Question: {question}\n ### Answer: {good_answer}"

        tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=max_new_tokens)
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

        my_question = f"### Question: {question}"
        tokenized = tokenizer(my_question)
        data["q_input_id_lens"].append(len(tokenized["input_ids"]))

        my_question = f"### Question: "
        tokenized = tokenizer(my_question)
        data["prompt_input_id_lens"].append(len(tokenized["input_ids"]))


    dataset = Dataset.from_dict(data)
    compression_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    compression_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=compression_data_collator, shuffle=True
    )
    if test_data:
        return compression_dataloader, data
    return compression_dataloader


def create_ib_attack_dataloader_ppo(protect_tokenizer, target_tokenizer, file_path="./data/jailbreaking_vicuna.csv", batch_size=4, max_new_tokens=400):
    """
    Create the advbench dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of advbench normal Q&A pairs.
    """
    df = pd.read_csv(file_path)
    df = df.iloc[120:]  # the frist 120 is the test data
    questions, ori_questions = df['attack_prompt'].to_list(), df["orig_goal"].to_list()
    good_answers = df["normal_response"].to_list()
    data = {"input_ids": [], "attention_mask": [], "q_input_ids": [],
     "q_attention_mask": [], "q_sub_input_ids": [], "q_sub_attention_mask": []}
    predict_data = {"input_ids": [], "attention_mask": [], "a_input_ids": [],
     "a_attention_mask": [], "q_sub_input_ids": [], "q_sub_attention_mask": []}

    for question, ori_question, good_answer in zip(questions, ori_questions, good_answers):
        #compression
        # text = f"### Summarize the follow sentence into a brief sentence, and end with '?' if the sentence is a question: {question}\n ### Summarization: {ori_question}"
        text = f"### Question: {question}\n ### Answer: {good_answer}"

        tokenized = protect_tokenizer(text, truncation=True, padding="max_length", max_length=max_new_tokens)
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

        my_question = f"### Question: {question}\n ### Answer: "
        tokenized = protect_tokenizer(my_question)
        data["q_input_ids"].append(tokenized["input_ids"])
        data["q_attention_mask"].append(tokenized["attention_mask"])

        tokenized = protect_tokenizer(ori_question)
        data["q_sub_input_ids"].append(tokenized["input_ids"])
        data["q_sub_attention_mask"].append(tokenized["attention_mask"])

        #prediction
        text2 = f"### Question: {ori_question}\n ### Answer: {good_answer}"
        tokenized2 = target_tokenizer(text2, truncation=True, padding="max_length", max_length=max_new_tokens, padding_side="right")
        predict_data["input_ids"].append(tokenized2["input_ids"])
        predict_data["attention_mask"].append(tokenized2["attention_mask"])

        tokenized2 = target_tokenizer(ori_question)
        predict_data["q_sub_input_ids"].append(tokenized2["input_ids"])
        predict_data["q_sub_attention_mask"].append(tokenized2["attention_mask"])

        tokenized2 = target_tokenizer(good_answer)
        predict_data["a_input_ids"].append(tokenized2["input_ids"])
        predict_data["a_attention_mask"].append(tokenized2["attention_mask"])


    dataset = Dataset.from_dict(data)
    predict_data = Dataset.from_dict(predict_data)
    compression_data_collator = DataCollatorForLanguageModeling(tokenizer=protect_tokenizer, mlm=False)
    prediction_data_collator = DataCollatorForLanguageModeling(tokenizer=target_tokenizer, mlm=False)

    compression_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=compression_data_collator, shuffle=False
    )
    prediction_dataloader = torch.utils.data.DataLoader(
        predict_data, batch_size=batch_size, collate_fn=prediction_data_collator, shuffle=False
    )
    return compression_dataloader, prediction_dataloader


def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):

    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)

    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    train_len = int(0.8 * len(all_ans))
    test_len = len(all_ans) - train_len

    questions, good_answers = df["Question"].values, df["Best Answer"].values
    # print(len(questions), len(good_answers))
    return all_ans



if __name__ == '__main__':
    all_ans = get_truthfulQA_answers_plaintext()
    print(len(all_ans))