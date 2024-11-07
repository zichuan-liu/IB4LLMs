Add datasets here

1. ### EasyJaibreak_results, you can download them in [EasyJailbreak](https://drive.google.com/file/d/1Im3q9n6ThL4xiaUEBmD7M8rkOIjw8oWU/view?usp=sharing).

2. GCG's Jailbreaking results, you can run the code from this [Repo](https://github.com/llm-attacks/llm-attacks/tree/main).

3. PAIR's Jailbreaking results, you can run the code from this [Repo](https://github.com/patrickrchao/JailbreakingLLMs).

4. TruthfulQA

Our directory format is

```
data/
│
├── EasyJaibreak_results/
│   ├── Autodan/
│   └── Cipher/
│   └── ......
│
├── GCG_new/
│   ├── individual_behavior_controls_llama2.json
│   └── individual_behavior_controls_vicuna.json
│   └── MaliciousInstruct_individual_behavior_controls_vicuna.json
│
├── orig_data/
│   ├── MaliciousInstruct.txt
│   └── advbench.txt
│
└── TruthfulQA.csv
└── jailbreaking_llama-2.csv
└── jailbreaking_vicuna.csv
```

You can change this format, but remember to change the path in the code `main.py`. *The form of the data is the same as that of the original author, and examples are given in the corresponding file.* Our dataset is described in Appendix D, which includes 520 instances per red-team.