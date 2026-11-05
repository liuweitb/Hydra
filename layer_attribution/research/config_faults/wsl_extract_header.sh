#!/usr/bin/env bash
# usage: wsl_extract_header.sh INFO_LOG OUT   copy the config echo (everything up to the first input packet)
LOG=$1; OUT=$2
n=$(grep -n -m1 "Got input @" "$LOG" | cut -d: -f1)
head -n "$n" "$LOG" | tr -d '\r' > "$OUT"
wc -c -l "$OUT"
