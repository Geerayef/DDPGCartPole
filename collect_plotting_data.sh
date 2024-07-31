#!/usr/bin/env bash

session=0

until [ $session == 5 ]; do
  python3 -m ddpg.cartpole | awk '/~~~~~ Start of results:/,/~~~~~ End of results./' >>session_results_per_episode.txt
  ((session++))
done

echo "~~~~~ Done \($session\)."
echo "~~~~~ Data collection complete."
