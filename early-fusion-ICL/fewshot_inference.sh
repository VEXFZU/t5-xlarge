#!/bin/bash

python fewshot_inference.py \
--model_name "SabaPivot/t5-xlarge-ko-kb-2" \
--revision "checkpoint-75000" \
--fewshot_path "fewshot-examples.txt" \
--target_input "삶과 죽음을 넘나드는 험난한 인생에서"