#!/bin/bash

# name of the input file
input_file="inp_file.txt"
n=3000
max_num_processes=12

# name of the output JSON file
output_file="timing_results.json"
echo "[" > $output_file

for np in $(seq 1 $max_num_processes)
do
  echo "Running with -np $np"
  output=$(mpiexec -np $np --use-hwthread-cpus ./a.out $input_file)
  elapsed_time=$(echo "$output" | grep -oP '(?<=max elapsed time is )\d+\.\d+s')
  json_entry="{\"n\": $n, \"processors\": $np, \"elapsed_time\": \"$elapsed_time\"}"

  if [ $np -ne 1 ]; then
    echo "," >> $output_file
  fi

  echo "$json_entry" >> $output_file
done

echo "]" >> $output_file

echo "Results saved to $output_file"
