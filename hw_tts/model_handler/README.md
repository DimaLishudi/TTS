# Homework 3 (TTS)

## Model handlers submodule

This submodule implements **TTSGenerator** and **TTSTrainer** classes. Both contain FastSpeech1/2 and WaveGlow models.\
TTSGenerator is used for audio synthesis and is used in *synthesis.py*.\
TTSTrainer inherits from it and also implements model training; it is used in *train.py*.
