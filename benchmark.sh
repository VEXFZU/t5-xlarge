#!/bin/bash

python benchmark.py \
--model_name "SabaPivot/t5-xlarge-ko-kb-2" \
--revision "checkpoint-75000" \
--benchmark_path "benchmark.json"