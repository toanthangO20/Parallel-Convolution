#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <runs> <command> [args...]"
  echo "Example: $0 5 ./seq/seq_conv.exe waterfall_grey_1920_2520.raw 1920 2520 50 grey"
  exit 1
fi

runs="$1"
shift

sum=0
for i in $(seq 1 "$runs"); do
  out="$($@)"
  # Expect the program to print a single number (seconds)
  val="$(printf '%s\n' "$out" | tail -n 1)"
  printf "Run %d: %s\n" "$i" "$val"
  sum=$(awk -v s="$sum" -v v="$val" 'BEGIN { printf "%.6f", s + v }')
  sleep 0.1
done

avg=$(awk -v s="$sum" -v n="$runs" 'BEGIN { printf "%.6f", s / n }')
printf "Average (%d runs): %s\n" "$runs" "$avg"