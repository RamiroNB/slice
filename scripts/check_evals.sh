#!/bin/bash

# Script to check which variant directories have all 5 checkpoint evals completed
# A checkpoint eval is considered completed when parallel_eval.log exists and does
# NOT contain "ModuleNotFoundError: No module named 'cl_lora'" (which marks a failed run)

check_variant_evals() {
    local variant_dir="$1"

    # Check if stages directory exists
    if [ ! -d "$variant_dir/stages" ]; then
        echo "0"
        return 1
    fi

    # Count how many stages have a valid parallel_eval.log
    # (file exists and does NOT contain the ModuleNotFoundError)
    local completed=0
    for stage_dir in "$variant_dir/stages"/stage_*; do
        local log="$stage_dir/parallel_eval.log"
        if [ -f "$log" ] && ! grep -q "ModuleNotFoundError: No module named 'cl_lora'" "$log"; then
            ((completed++))
        fi
    done

    echo "$completed"
    [ $completed -eq 5 ]
}

check_folder() {
    local folder="$1"
    local folder_name=$(basename "$folder")

    echo "========== $folder_name =========="
    local complete=0
    local incomplete=0

    echo "-- Complete (5/5) --"
    for variant in "$folder"/*/; do
        count=$(check_variant_evals "$variant")
        if [ "$count" = "5" ]; then
            echo "  ✓ $(basename "$variant")"
            ((complete++))
        fi
    done

    echo ""
    echo "-- Incomplete --"
    for variant in "$folder"/*/; do
        count=$(check_variant_evals "$variant")
        if [ "$count" != "5" ]; then
            echo "  ✗ $(basename "$variant") ($count/5)"
            ((incomplete++))
        fi
    done

    echo ""
    echo "Complete: $complete | Incomplete: $incomplete"
    echo ""
}

check_folder /mnt/E-SSD/dev-cl-lora/cl-lora/results/NI-Seq-Opposite-v2
check_folder /mnt/E-SSD/dev-cl-lora/cl-lora/results/NI-Seq-Opposite-v4
check_folder /mnt/E-SSD/dev-cl-lora/cl-lora/results/NI-Seq-Opposite-v3
