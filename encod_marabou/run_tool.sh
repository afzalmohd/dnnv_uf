#!/usr/bin/bash

# Check if at least two arguments are passed (num_cpu and config_file)
if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_cpu> <log_dir>"
    exit 1
fi

# # Assign command-line arguments to variables
num_cpu=$1
# config_file=$2
# python script.py $config_file
log_dir=$2

# Set permissions for the log directory
chmod -R u+x $log_dir/

# Launch script instances for each CPU
for ((i=0; i<$num_cpu; i++))
do
    $log_dir"/script_$i.sh" &
done
