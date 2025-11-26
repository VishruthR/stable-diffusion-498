#!/bin/bash

# Array of strength values to test
strengths=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Common parameters
PROMPT="a hyperrealistic photograph of a cat with random eye color"
N_SAMPLES=9
MAX_BATCH_SIZE=3
WIDTH=512
HEIGHT=512
DDIM_STEPS=100
BASE_OUTDIR="outputs/test_strengths"

# Create base output directory
mkdir -p "$BASE_OUTDIR"

for strength in "${strengths[@]}"
do
    echo "----------------------------------------------------------------"
    echo "Running test with strength: $strength"
    echo "----------------------------------------------------------------"
    
    # Create specific output directory for this strength
    CURRENT_OUTDIR="${BASE_OUTDIR}/strength_${strength}"
    
    python scripts/batch_gen.py \
        --prompt "$PROMPT" \
        --n_samples $N_SAMPLES \
        --max_batch_size $MAX_BATCH_SIZE \
        --W $WIDTH \
        --H $HEIGHT \
        --strength $strength \
        --ddim_steps $DDIM_STEPS \
        --outdir "$CURRENT_OUTDIR"
        
    echo "Finished strength $strength"
    echo ""
done
