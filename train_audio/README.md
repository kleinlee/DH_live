# LSTM Training Notes

## Data Preparation
- Use over 10,000 short audio clips with diverse speakers, each at least 5 seconds long, in 16k single-channel wav format.
- Place these files in the `train_audio/train_data` folder.
- Download `wav2lip.pth` from the wav2lip project (https://github.com/Rudrabha/Wav2Lip) and place it in the `train_audio/checkpoints` folder.

#### Notes
Audio files must be in 16k single-channel wav format.

## Steps

### 1. Face Video Generation
```bash
python preparation_step0.py <face_path> <wav_16K_path>
# Example: preparation_step0.py face.jpg train_data
```

### 2. Mouth Region Extraction and PCA Modeling
```bash
python preparation_step1.py <data_path>
# Example: preparation_step1.py train_data
```

Now ensure the file directory is as follows:
```bash
|--/train_audio
|  |--/checkpoints
|  |  |--/wav2lip.pth
|  |  |--/pca.pkl
|  |  |--/wav2lip_pca_all.gif
|  |--/train_data
|  |  |--/000001.wav
|  |  |--/000001.avi
|  |  |--/000001.txt
|  |  |--/000002.wav
|  |  |--/000002.avi
|  |  |--/000002.txt
|  |  |--/000003.wav
|  |  |--/000003.avi
|  |  |--/000003.txt
```

### 3. Training LSTM Model
```bash
python train_lstm.py <data_path>
# Example: train_lstm.py train_data
```

### 4. Testing Audio Accuracy
```bash
python test.py <wav_path> <ckpt_path>
# Example: python test.py D:/Code/py/test_wav/0013.wav checkpoints/audio.pkl
```


