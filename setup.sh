#!/bin/bash

# install requirements
pip install -qq -r requirements.txt
apt-get install -y -qq wget unzip

# prepare folders
mkdir data

#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2 

# gdown --no-cookies https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
# mv train.txt data/


# #download Waveglow
# gdown --no-cookies https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
# mv ./waveglow_256channels_ljs_v2.pt ./hw_tts/FS_utils/waveglow/pretrained_model/waveglow_256channels.pt

# gdown --no-cookies https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
# tar -xvf mel.tar.gz
# mv mels ./data/mels

# #download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip -d ./data >> /dev/null
rm alignments.zip

# preprocess data
python3 preprocess_data.py


# we will use waveglow code, data and audio preprocessing from this repo
# git clone https://github.com/xcmyz/FastSpeech.git
# mv FastSpeech/text ./FS_utils/
# mv FastSpeech/audio ./FS_utils/
# mv FastSpeech/waveglow/* ./FS_utils/waveglow/
# mv FastSpeech/utils.py ./FS_utils/
# mv FastSpeech/glow.py ./FS_utils/
# rm -rf FastSpeech