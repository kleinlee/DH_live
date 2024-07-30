import torch.nn as nn
import torch
class Audio2Feature(nn.Module):
    def __init__(self):
        super(Audio2Feature, self).__init__()
        num_pred = 1
        self.output_size = 6
        self.ndim = 80
        APC_hidden_size = 80
        # define networks
        self.downsample = nn.Sequential(
            nn.Linear(in_features=APC_hidden_size * 2, out_features=APC_hidden_size),
            nn.BatchNorm1d(APC_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(APC_hidden_size, APC_hidden_size),
        )
        self.LSTM = nn.LSTM(input_size=APC_hidden_size,
                            hidden_size=192,
                            num_layers=2,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=192, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.output_size))

    def forward(self, audio_features, h0, c0):
            '''
                Args:
                audio_features: [b, T, ndim]
            '''
            self.item_len = audio_features.size()[1]
            # new in 0324
            audio_features = audio_features.reshape(-1, self.ndim * 2)
            down_audio_feats = self.downsample(audio_features)
            # print(down_audio_feats)
            down_audio_feats = down_audio_feats.reshape(-1, int(self.item_len / 2), self.ndim)
            output, (hn, cn) = self.LSTM(down_audio_feats, (h0, c0))

            #            output, (hn, cn) = self.LSTM(audio_features)
            pred = self.fc(output.reshape(-1, 192)).reshape(-1, int(self.item_len / 2), self.output_size)
            return pred, hn, cn
