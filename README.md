# Homework 3 (TTS)

Implements FastSpeech1/2 models via PyTorch. [Wandb report](https://wandb.ai/dlishudi/TTS/reports/Text-To-Speech--VmlldzozMDUxMDU4)

## Project Navigation

* hw_tts/ is main module of this project; it directly implements FastSpeech and FastSpeech2.
* requirements.txt lists all python requirements.
* preprocess_data.py implements data preprocessing.
* setup.sh script installs needed requirements, downloads data and preprocesses it.
* train.py and synthesis.py scripts are used to train model and synthesise audio respectively.
* launcher.ipynb notebook shows possible ways to run scripts for this project.
* figs/ directory contains some figures for report.

## Requirements and Set Up
To install all requirements, download needed data and preprocess it simply run:
```
./setup.sh
```

## Model Training and Audio Synthesis
To train model run 
```
python3 ./train.py -c ./hw_tts/configs/fs2_train_config.json
```
To synthesise audio run
```
python3 ./synthesis.py -c ./hw_tts/configs/fs2_synthesis_config.json -w ./fs2_saved/checkpoint_19999.pth.tar
```

All scripts menthioned above are also shown in launcher.ipynb notebook.

## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [ming024's FastSpeech2 implementation](https://github.com/ming024/FastSpeech2)
