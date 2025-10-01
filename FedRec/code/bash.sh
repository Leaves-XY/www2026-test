#conda activate py39_torch

bsub -N -q volta \
-e ./lsflog/%J.err \
-o ./lsflog/%J.out \
-n 1 \
-gpu "num=1:mode=exclusive_process" \
"python experiment.py >> ./lsflog/out.log"