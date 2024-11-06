#!/usr/bin/bash

# Check if at least two arguments are passed (num_cpu and config_file)
if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_cpu> <config_file>"
    exit 1
fi

# Assign command-line arguments to variables
num_cpu=$1
config_file=$2

# Set up directories
root_dir=`pwd`
log_dir=$root_dir/logs

# Remove existing logs and create new log directory
rm -rf $log_dir
mkdir -p $log_dir

# result_dir=$root_dir/results
# result_dir=$result_dir/mnistdeeppoly.csv
# export RESULT_FILE=$result_dir

# Run Python script with provided arguments
python script.py $num_cpu $log_dir $config_file

# Set permissions for the log directory
chmod -R u+x $log_dir/

# Launch script instances for each CPU
for ((i=0; i<$num_cpu; i++))
do
    $log_dir"/script_$i.sh" &
done
