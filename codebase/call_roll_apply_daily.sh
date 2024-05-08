#!/bin/bash
for value in "rvol" "rmax" "skew" "capm_ext" "ff3" "hxz4" "dimsonbeta" "zero_trades"; do
    python roll_apply_daily.py 15 _21d "$value"
done
for value in "zero_trades" "turnover" "dolvol" "ami"; do
    python roll_apply_daily.py 60 _126d "$value"
done
for value in "rvol" "capm" "downbeta" "zero_trades" "prc_to_high" "mktvol"; do
    python roll_apply_daily.py 120 _252d "$value"
done
for value in "mktcorr"; do
    python roll_apply_daily.py 750 _1260d "$value"
done
# Exit with the exit status of the Python script
exit $?
