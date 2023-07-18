#!/usr/bin/env bash

filename=0

while IFS= read -r line; do
    # echo "Text read from file: $line"
    if [ "$line" == "~~~~~ Start of results:" ]
    then
        echo "Parsing plot data for line $filename"
    elif [ "$line" == "~~~~~ End of results." ]
    then
        (( filename++ ))
    else
        echo $line >> LineData/Line$filename.txt
    fi
done < "$1"
