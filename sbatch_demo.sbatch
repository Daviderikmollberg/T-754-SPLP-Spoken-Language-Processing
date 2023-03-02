#!/usr/bin/env bash

#SBATCH --mem 200MB
#SBATCH --cpus-per-task 2
#SBATCH --output ./"%x-%j.log"
#SBATCH --time 0-07:00:00

echo "Hello world"
sleep 15
echo "Goodbye world"



