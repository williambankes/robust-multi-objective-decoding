#!bin/bash

OPENAI_API_KEY=""
NUM_PROMPTS=100

COL1="response (before)"
python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/eval_outputs/HH/eval_hh_average_bs1024_branch2-4-8-16__depth16_topk-1_gsTrue.pkl" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_reference_${NUM_PROMPTS}prompts.csv" \
    --col1 "${COL1}" \
    --num_prompts ${NUM_PROMPTS}

COL1="response harmless (1.000-16)"
python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/eval_outputs/HH/eval_hh_harmless_bs1024_branch2-4-8-16__depth16_topk-1_gsTrue.pkl" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_harmless_blockwise_K16B16_${NUM_PROMPTS}prompts.csv" \
    --col1 "${COL1}" \
    --num_prompts ${NUM_PROMPTS}

COL1="response helpful (1.000-16)"
python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/eval_outputs/HH/eval_hh_helpful_bs1024_branch2-4-8-16__depth16_topk-1_gsTrue.pkl" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_helpful_blockwise_K16B16_${NUM_PROMPTS}prompts.csv" \
    --col1 "${COL1}" \
    --num_prompts ${NUM_PROMPTS}

COL1="response (0.500-16)"
python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/eval_outputs/HH/eval_hh_robust_bs1024_branch2-4-8-16_vcoef0.5_depth16_topk-1_gsTrue.pkl" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_robust_blockwise_vcoef0.5_K16B16_${NUM_PROMPTS}prompts.csv" \
    --col1 "${COL1}" \
    --num_prompts ${NUM_PROMPTS}

COL1="response average (1.000-16)"
python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/eval_outputs/HH/eval_hh_average_bs1024_branch2-4-8-16__depth16_topk-1_gsTrue.pkl" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_uniform_blockwise_K16B16_${NUM_PROMPTS}prompts.csv" \
    --col1 "${COL1}" \
    --num_prompts ${NUM_PROMPTS}

python3 run_llm_single_eval.py \
    --openai_api_key ${OPENAI_API_KEY} \
    --data_path1 "/home/hingdoong/0_codes/safe-decoding/results_eval_hh/hh_results__home_ubuntu_virginia0_safe-decoding_checkpoints_sft_HH_gemma-2-2b-it_RMOD_lr1e-5_3epochs_16000prompts_checkpoint-1494_temp1.0_topk50_topp1.0_tokens256_num_samples1024.csv" \
    --result_dir "/home/hingdoong/0_codes/robust-multi-objective-decoding/results_llm_single_eval/" \
    --result_name "gpt4o_RMOD_distillation_K16B16_${NUM_PROMPTS}prompts.csv" \
    --num_prompts ${NUM_PROMPTS}
