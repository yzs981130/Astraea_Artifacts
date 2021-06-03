cd simulation
python main.py --block_scheduling 1 --num_node_p_switch 140 --num_switch 1 --stavation_threshold 21600 --delay_threshold 300 --schedule lease --placement consolidate --job_selection fairness --log_level ERROR --trace ../evaluation/modified-philly-trace/philly_two_weeks.csv --share ../evaluation/modified-philly-trace/share-production.csv --name_list ../evaluation/modified-philly-trace/name-production.lst --lease_term_interval 900 --metrics True --metrics_path 21600-300-de94e51-lease-consolidate-job_reward-fair-900-metrics.csv --user_quota fair --dist_trace_path data/sacct_sh40.csv --fairness_output 21600-300-de94e51-lease-consolidate-job_reward-fair-900-result.csv > 21600-300-de94e51-lease-consolidate-job_reward-fair-900.log
