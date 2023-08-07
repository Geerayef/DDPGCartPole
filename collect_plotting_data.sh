#!/usr/bin/env bash

session=0
conda_env="tf"

# _check_conda_env () {
#     trim() {
#         local var="$*"
#         var="${var#"${var%%[![:space:]]*}"}"
#         var="${var%"${var##*[![:space:]]}"}"
#         echo -n "$var"
#     }
#
#     if [ -e ~/miniconda3 ]
#     then
#         echo "~~~~~ Checking conda environment"
#         conda_env=$(conda info -vq | awk -F: '{print $2}' | grep -E -m 1 "base|tf|None") 
#         # echo "Conda env =" $conda_env
#         conda_env=$(trim $conda_env)
#         if [ "$conda_env" == "tf" ]
#         then
#             echo "~~~~~ OK: conda environment \"tf\" is active"
#             # echo $conda_env
#         elif [ "$conda_env" == "base" ]
#         then
#             echo "~~~~~ WARNING: conda environment \"base\" is active. Activate \"tf\" environment!"
#             # echo $conda_env
#         else
#             echo "~~~~~ ERROR: No active conda environments."
#             # echo $conda_env
#         fi
#     else
#         echo "~~~~~ Could not find ~/miniconda3"
#     fi
# }
#
# _check_conda_env 

if [ "$conda_env" == "tf" ]
then
    echo "Good."
    until [ $session == 5 ]
    do
        python3 -m ddpg.cartpole | awk '/~~~~~ Start of results:/,/~~~~~ End of results./' >> session_results_per_episode.txt
        (( session++ ))
    done
fi

echo "~~~~~ Done \($session\)."
echo "~~~~~ Data collection complete."
