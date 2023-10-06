#! /bin/bash

files=(outputs lightning_logs)

for file in ${files[@]}; do
    if [ -d $file ]; then
        echo "Removing $file"
        rm -r $file
    fi
done