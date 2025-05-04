#!/bin/bash

# Set the start and end of the range
START=33
END=75

# Loop through the range
for i in $(seq $START $END); do
    echo "Processing g5/$i.pdf"
    python run/pdf_ocr.py g5/$i.pdf
done