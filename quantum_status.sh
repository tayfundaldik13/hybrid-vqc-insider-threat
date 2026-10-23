#!/usr/bin/env bash
cd "$(dirname "$0")"
D=results_3/quantum/folds
echo "=== $(date '+%H:%M:%S') ==="
R=$(pgrep -f run_parallel_quantum.sh | head -1)
[ -n "$R" ] && echo "runner elapsed : $(ps -o etime= -p $R | tr -d ' ')" || echo "runner         : NOT running"
echo "workers running: $(pgrep -fc '^/.*python parallel_quantum.py worker')"
TOTAL=$(ls logs_parallel_quantum/seed*_stage*_fold*.log 2>/dev/null | wc -l)
echo "folds finished : $(ls $D/*.json 2>/dev/null | wc -l)   (logs started: $TOTAL)"
echo "errors         : $(grep -l 'Traceback' logs_parallel_quantum/*.log 2>/dev/null | wc -l) log(s) with Traceback"
echo "--- finished folds ---"
for f in $D/*.json; do [ -e "$f" ] || { echo "(none yet)"; break; }
  python3 - "$f" <<'P'
import json,sys,os
d=json.load(open(sys.argv[1])); m=d["metrics"]
print(f"{os.path.basename(sys.argv[1])[:-5]:<24} F1={m['f1']:.3f} MacroF1={m['macro_f1']:.3f} AUC={m['auc']:.3f} "
      f"optuna={d['opt_time_s']/3600:.2f}h train={d['fold_time_s']/3600:.2f}h")
P
done
