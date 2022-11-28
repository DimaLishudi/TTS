# Homework 3 (TTS)

## Main submodule (hw_tts)

This module consist of following submodules:
* **FS_utils** -- text and audio preprocessing utils and WaveGlow vocoder from [FastSpeech repository by xcmyz](https://github.com/xcmyz/FastSpeech).
* **configs** -- directory containing configs for train and synthesis.
* **dataset** -- dataset building submodule.
* **logger** -- Wandb logger submodule.
* **loss** -- FastSpeech1 and FastSpeech2 Loss classes submodules.
* **model** -- pytorch implementation of FastSpeech1 and FastSpeech2 models.
* **model_handler** -- wrappers around FastSpeech1/2 model implementing train and synthesis.
