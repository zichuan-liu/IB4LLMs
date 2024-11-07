# <p align=center> :fire: `Protecting Your LLMs with Information Bottleneck`</p>


[[Arxiv Paper](https://arxiv.org/abs/2404.13968)] [[Slides](https://zichuan-liu.github.io/talk/ib_slides.pdf)] [[中文版](https://zhuanlan.zhihu.com/p/694129510)] [[Website Page](https://zichuan-liu.github.io/projects/IBProtector/index.html)] 

**TL;DR**: We propose IBProtector, the first LLM jailbreak defending method based on the Information Bottleneck principle. Our protector efficiently defends against adversarial prompts without losing key information.

![figs](figs/framework.png)

> 🌟 If you want to read the main code of IBProtector, check out the class **VIBLLM** in the file `./lib/defenses.py`.

## Dependent version

```bash
pip install -U datasets==2.14.5 torch==2.1.1 torchmetrics==1.2.0 bitsandbytes==0.43.0 openai==0.28.0 fschat==0.2.20
pip install -U transformers==4.40.1
```
``please don't change the order, otherwise it will have comfict sicne this fschat verson is too old``

## Getting Started

### model configuration
First set up the path and parameters of your LLMs configuration in the file `lib/model_configs.py`


### data preparing
Datasets acquisition instuction in the file `./data/README.md`, overall, you should get your jailbreaking data by `GCG` or `PAIR` in advance. 

### how to tune models

To finetune the IBProtector of Vicuna-13b, you can execute the following command
```bash
python test_finetuning.py
```

To finetune the IBProtector of Llama2-7b, you can execute the following command
```bash
python test_finetuning_llama.py
```

The baselines `sft` and `unlearning` are set in files `test_sft.py` and `test_unlearning.py`, respectively.

### how to inference

You can execute the following command
```bash
python main.py  --results_dir ./our_results  --target_model vicuna  --attack TriviaQA --method vib --cuda 0
```
The denfense **method** can be chosen: `none`, `smooth`, `selfdefense`, `sft`, `unlearning`, `ra`, `semantic`, and `vib`, Note that the `vib`, `sft`, and `unlearning` need to fine-tune the LLMs in advance, via the corresponding commands. 

The **attack** method can be chosen: `GCG` and `PAIR` for the main experiment, `EasyJailbreak` for the transferability,  and `TriviaQA` for testing benign answering rates.


You can also run the main results through the script:

```bash
bash script.sh
```

## Evaluating the results

Please find the examples in `./eval/eval_asr.py`,  `./eval/eval_harm.py`, `./eval/eval_gpt.py`, `./eval/eval_friedman.py`, and `./eval/eval_time.py` to evaluate the results. The main you need is to change your result path by modifying `file_path`


For instance, the evaluating command is:
```bash
cd eval/
python eval_asr.py  --file_path `YOUR_RESULTS_PATH`
```



## Further Reading
For more information about theories and limitations of existing perturbation methods, please see [THIS SLIDE](https://zichuan-liu.github.io/talk/ib_slides.pdf).

The following are related works:

1, [**Explaining Time Series via Contrastive and Locally Sparse Perturbations**](https://openreview.net/pdf?id=qDdSRaOiyb), in ICLR 2024.
[\[GitHub Repo\]](https://github.com/zichuan-liu/ContraLSP)


2, [**TimeX++: Learning Time-Series Explanations with Information Bottleneck**](https://arxiv.org/abs/2405.09308), in ICML 2024.
[\[GitHub Repo\]](https://github.com/zichuan-liu/TimeXplusplus)


## Citing IBProtector
🌟 If you find this resource helpful, please consider starting this repository and cite our research:
```tex
@inproceedings{liu2024protecting,
      title={Protecting Your LLMs with Information Bottleneck}, 
      author={Zichuan Liu and Zefan Wang and Linjie Xu and Jinyu Wang and Lei Song and Tianchun Wang and Chunlin Chen and Wei Cheng and Jiang Bian},
      year={2024},
      booktitle={Neural Information Processing Systems}
}
```
In case of any questions, bugs, suggestions, or improvements, please feel free to drop me at _zichuanliu@smail.nju.edu.cn_ or open an issue.
