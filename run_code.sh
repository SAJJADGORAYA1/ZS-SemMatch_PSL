#!/bin/bash

# Define lists for parameters
seed_list=(1)
# method_list=(c3d cnn_lstm mhi_baseline mhi_fusion mhi_attention mediapipe mediapipe_transformer mediapipe_lstm c3d_pretrained cnn_lstm_pretrained finetuned_gemini zero_shot semantic_shot)
method_list=(cnn_lstm mhi_baseline mhi_fusion mhi_attention mediapipe mediapipe_transformer mediapipe_lstm)
num_words_list=(30)
epochs_list=(100)
batch_size_list=(4)

# Loop over parameters
for seed in "${seed_list[@]}"; do
    for epochs in "${epochs_list[@]}"; do
        for batch_size in "${batch_size_list[@]}"; do
            for method in "${method_list[@]}"; do
                for num_words in "${num_words_list[@]}"; do
                    echo "Running: python3 main.py --method $method --num_words $num_words --seed $seed --epochs $epochs --batch_size $batch_size"
                    
                    # Add --no_pretrained flag for methods that need it
                    if [[ "$method" == "c3d" || "$method" == "cnn_lstm" || "$method" == "mhi_fusion" || "$method" == "mhi_attention" ]]; then
                        python3 main.py --method "$method" --num_words "$num_words" --seed "$seed" --epochs "$epochs" --batch_size "$batch_size" --no_pretrained --confusion --loss
                    elif [[ "$method" == "c3d_pretrained" || "$method" == "cnn_lstm_pretrained" ]]; then
                        python3 main.py --method "$method" --num_words "$num_words" --seed "$seed" --epochs "$epochs" --batch_size "$batch_size" --confusion --loss
                    else
                        python3 main.py --method "$method" --num_words "$num_words" --seed "$seed" --confusion
                    fi
                    
                    echo "Completed: $method with seed $seed, num_words $num_words, epochs $epochs, batch_size $batch_size"
                    echo "----------------------------------------"
                done
            done
        done
    done
done

echo "All methods completed!"
