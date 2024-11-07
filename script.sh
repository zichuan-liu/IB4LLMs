python test_finetuning.py
python test_finetuning_llama.py

##ablation

python test_finetuning.py --ab_mode True --lamda 1 --alpha 2 --r 0.1
python test_finetuning.py --ab_mode True --lamda 1 --alpha 2 --r 0.3
python test_finetuning.py --ab_mode True --lamda 1 --alpha 2 --r 0.5
python test_finetuning.py --ab_mode True --lamda 1 --alpha 2 --r 0.7
python test_finetuning.py --ab_mode True --lamda 1 --alpha 2 --r 0.9


python main.py  --results_dir ./benign_results  --target_model vicuna --attack TriviaQA --method vib --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method vib --cuda 0
python main.py  --results_dir ./benign_results  --target_model vicuna --attack TriviaQA --method none --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method none --cuda 0
python main.py  --results_dir ./benign_results  --target_model vicuna --attack TriviaQA --method selfdefense --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method selfdefense --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method unlearning --cuda 0 #
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method sft --cuda 0
python main.py  --results_dir ./benign_results  --target_model vicuna --attack TriviaQA --method smooth --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method smooth --cuda 0
python main.py  --results_dir ./benign_results  --target_model vicuna --attack TriviaQA --method ra --cuda 0
python main.py  --results_dir ./benign_results  --target_model llama2 --attack TriviaQA --method ra --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method sft --cuda 0 #A100 OOM
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack GCG --method sft --cuda 0  #A100 OOM
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method sft --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method sft --cuda 0


python main.py  --results_dir ./repeat_results  --target_model llama2 --attack GCG --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model llama2 --attack PAIR --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack GCG --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack PAIR --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack TriviaQA --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model llama2 --attack TriviaQA --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack EasyJailbreak --method semantic --cuda 0
python main.py  --results_dir ./repeat_results  --target_model llama2 --attack EasyJailbreak --method semantic --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method unlearning --cuda 0 #A100 OOM
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack GCG --method unlearning --cuda 0  #A100 OOM
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method unlearning --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method unlearning --cuda 0

python main.py  --results_dir ./repeat_results  --target_model llama2 --attack PAIR --method none --cuda 0
python main.py  --results_dir ./repeat_results  --target_model llama2 --attack GCG --method none --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack PAIR --method none --cuda 0
python main.py  --results_dir ./repeat_results  --target_model vicuna --attack GCG --method none --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method selfdefense --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method selfdefense --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method selfdefense --cuda 0
python main.py  --results_dir ./repeat_results2 --target_model vicuna --attack GCG --method selfdefense --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method smooth --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method smooth --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method smooth --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack GCG --method smooth --cuda 0


python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method ra --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method ra --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method ra --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack GCG --method ra --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack PAIR --method vib --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack GCG --method vib --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack PAIR --method vib --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack GCG --method vib --cuda 0

python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method unlearning --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method sft --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack EasyJailbreak --method selfdefense --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method selfdefense --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model vicuna --attack EasyJailbreak --method ra --cuda 0
python main.py  --results_dir ./repeat_results2  --target_model llama2 --attack EasyJailbreak --method ra --cuda 0
