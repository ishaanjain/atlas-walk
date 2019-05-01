#!/usr/bin/env bash

numargs=$#
model=$1

if [ $numargs -eq 0 ]
then
    echo "Error: provide the type of model to use"
elif [ $model == "DDPG" ]
then
    echo "Running humanoid environment with the deep deterministic policy gradient model..."
    python3 DDPG/main.py "${@:2:$numargs}"
elif [ $model == "CDQN" ]
then
    echo "Running humanoid environment with the continuous deep Q-learning model..."
    python3 CDQN/main.py ${@:2:$numargs}
else
    echo "Error: not a valid model type"
fi
