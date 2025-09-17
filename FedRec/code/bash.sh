bsub -N -q gpu \
-e ./lsflog/%J.err \
-o ./lsflog/%J.out \
-n 1 \
-gpu "num=1:mode=exclusive_process" \
"python experiment.py >> ./lsflog/out.log"