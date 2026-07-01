#!/usr/bin/env bash

# budgets=(0.003 0.033 0.1 0.333 1)

# for budget in "${budgets[@]}"; do
#     echo "Running with train_data_budget=${budget}"

#     CUDA_VISIBLE_DEVICES="0" python train_task_force.py \
#         +experiment=xela/task/force/dinov2.yaml \
#         ++train_data_budget="${budget}"
# done


budgets=(0.003 0.033 0.1 0.333 1)
LOG_DIR="$(pwd)"
pids=()

for budget in "${budgets[@]}"; do
    echo "Running with train_data_budget=${budget}"

    CUDA_VISIBLE_DEVICES="0" python train_task_force.py \
        +experiment=xela/task/force/dinov2.yaml \
        ++train_data_budget="${budget}" \
        > "${LOG_DIR}/train_budget_${budget}.log" 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "PID for budget ${budget}: ${pid}"
done

echo ""
echo "All jobs started in background. Logs saved to ${LOG_DIR}"
echo "To stop a job, use: kill <PID>"
echo "All PIDs: ${pids[*]}"
