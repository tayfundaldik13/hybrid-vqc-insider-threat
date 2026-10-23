#!/usr/bin/env bash
cd "$(dirname "$0")"
R=results_all/quantum/folds
L=logs_parallel_quantum

echo "=== $(date '+%F %T') ==="
echo "Running workers : $(pgrep -f 'parallel_quantum_all.py worker' | xargs -r ps -o args= -p 2>/dev/null | grep -c '^/.*python parallel_quantum_all.py worker\|^.*python .*parallel_quantum_all.py worker [0-9]')"
echo "Load average    :$(uptime | sed 's/.*load average://')"
free -g | awk 'NR==2{printf "RAM used        : %s / %s GB\n",$3,$2}'
echo
done_n=$(ls $R/seed*_fold*.json 2>/dev/null | wc -l)
echo "Folds finished  : $done_n / 40"
echo
printf "%-6s %s\n" seed "fold0 fold1 fold2 fold3 fold4   (+ = done, . = running)"
for s in 44 45 46 47 48 49 50 51; do
  line=""
  for f in 0 1 2 3 4; do
    if [ -f $R/seed${s}_fold${f}.json ]; then line="$line  +  "; else line="$line  .  "; fi
  done
  printf "%-6s %s\n" $s "$line"
done
echo
echo "--- finished fold results ---"
for j in $R/seed*_fold*.json; do
  [ -f "$j" ] || continue
  python3 - "$j" <<'E'
import json,sys,os
d=json.load(open(sys.argv[1])); m=d["metrics"]
n=os.path.basename(sys.argv[1])[:-5]
print(f"{n:14s} F1={m['f1']:.4f} MCC={m['mcc']:.4f} AUC={m['auc']:.4f} ECE={m['ece']:.4f}  "
      f"train={d['fold_time_s']/3600:.1f}h optuna={d['opt_time_s']/3600:.1f}h")
E
done
echo
echo "--- errors in logs ---"
grep -il "traceback\|error" $L/*.log 2>/dev/null | head || true
