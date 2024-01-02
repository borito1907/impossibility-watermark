TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python attack.py --input "./inputs/lotr.csv" --output "./results/output_$TIMESTAMP.csv" --step_T 500 --num_trials 1 --check_quality True --intermediate "./results/intermediate_$TIMESTAMP.csv" --result_stats "./results/stats_$TIMESTAMP.csv"