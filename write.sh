#!/bin/bash

for file in *.py; do
    if [ -f "$file" ]; then
        echo "======== START OF $file =========="
        cat "$file"
        echo "======== END OF $file =========="
        echo
    fi
done

