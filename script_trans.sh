python main.py  --results_dir ./t_results2  --target_model llama2 --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna7b --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results2  --target_model chatglm3 --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results2  --target_model mistral --multi True --attack EasyJailbreak --method none --cuda 0

python main.py  --results_dir ./t_results2  --target_model llama2 --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna7b --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results2  --target_model chatglm3 --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results2  --target_model mistral --multi True --attack EasyJailbreak --method vib --cuda 0

python main.py  --results_dir ./t_results2  --target_model llama2 --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results2  --target_model vicuna7b --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results2  --target_model chatglm3 --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results2  --target_model mistral --multi True --attack EasyJailbreak --method smooth --cuda 0

python main.py  --results_dir ./t_results2  --target_model llama2 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results2  --target_model vicuna --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results2  --target_model vicuna7b --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results2  --target_model chatglm3 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results2  --target_model mistral --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2

python main.py  --results_dir ./t_results2  --target_model llama2 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4
python main.py  --results_dir ./t_results2  --target_model vicuna --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4
python main.py  --results_dir ./t_results2  --target_model vicuna7b --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4
python main.py  --results_dir ./t_results2  --target_model chatglm3 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4
python main.py  --results_dir ./t_results2  --target_model mistral --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4

python main.py  --results_dir ./t_results  --target_model chatgpt --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results  --target_model chatgpt --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results  --target_model chatgpt --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results  --target_model chatgpt --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results  --target_model chatgpt --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4

python main.py  --results_dir ./t_results  --target_model gpt4 --multi True --attack EasyJailbreak --method none --cuda 0
python main.py  --results_dir ./t_results  --target_model gpt4 --multi True --attack EasyJailbreak --method vib --cuda 0
python main.py  --results_dir ./t_results  --target_model gpt4 --multi True --attack EasyJailbreak --method smooth --cuda 0
python main.py  --results_dir ./t_results  --target_model gpt4 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 2
python main.py  --results_dir ./t_results  --target_model gpt4 --multi True --attack EasyJailbreak --method smooth --cuda 0 --smoothllm_num_copies 4