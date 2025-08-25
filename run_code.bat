@echo off
setlocal

REM Define lists for parameters
set seed_list=1
set method_list=c3d c3d_pretrained cnn_lstm cnn_lstm_pretrained mhi_fusion mhi_attention mediapipe_transformer mediapipe_lstm
set num_words_list=1
set epochs_list=3
set batch_size_list=1

REM Loop over parameters
for %%s in (%seed_list%) do (
    for %%e in (%epochs_list%) do (
        for %%b in (%batch_size_list%) do (
            for %%m in (%method_list%) do (
                for %%n in (%num_words_list%) do (
                    echo Running: python3 main.py --method %%m --num_words %%n --seed %%s --epochs %%e --batch_size %%b
                    python3 main.py --method %%m --num_words %%n --seed %%s --epochs %%e --batch_size %%b
                )
            )
        )
    )
)

endlocal
