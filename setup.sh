#!/bin/bash

# install requirements
pip install -qq -r requirements.txt
apt-get install -y -qq wget unzip

# prepare folders
mkdir data
mkdir ./hw_tts/FS_utils/waveglow/pretrained_model

#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2 

gdown --no-cookies https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/


# download Waveglow
gdown --no-cookies https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mv ./waveglow_256channels_ljs_v2.pt ./hw_tts/FS_utils/waveglow/pretrained_model/waveglow_256channels.pt


# download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip -d ./data >> /dev/null
rm alignments.zip

# download checkpoint
mkdir ./fs2_saved
gdown --no-cookies https://drive.google.com/file/d/1N4LK-k1Ox4ToJlISpSf52oXYj5W4MRlk
mv ./checkpoint_19999.pth.tar ./fs2_saved/checkpoint_19999.pth.tar

# preprocess data
python3 preprocess_data.py