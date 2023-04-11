for i in {1..6}; do
    python gradient_analysis.py \
        --scale=$i \
        --logdir='Gradient_analysis/Float_precision'
done