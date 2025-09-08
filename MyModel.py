import torch
import pandas as pd
from torch import Tensor, nn
import numpy as np

CATEGORICAL_FEATURES = ['sex','season','emgdeptchin_name','dayzone','judegmentcode_new','maritalstatuscode_new','triage','diagnosisserious','ambulance','nojob']
NUMERIC_FEATURES = ['age','los','weekend','weekend_dischargetime','dayzone_discharge','houseincome','sex_vs','vs_age','taiwan',]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES



# ===================TCN=================== #
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
    
        # First conv block
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                            stride=stride, padding=(0, padding), dilation=(1, dilation))
        self.bn1 = nn.BatchNorm2d(n_outputs)
        
        # Second conv block
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                            stride=stride, padding=(0, padding), dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(n_outputs)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.dropout,
            self.conv2,
            self.bn2,
            self.relu,
            self.dropout
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # input shape = (batch_size, channels, 1, sequence_length)
        out = self.net(x).squeeze(2)  # 保持形狀 (batch_size, channels, sequence_length)
        res = x.squeeze(2) if self.downsample is None else self.downsample(x.squeeze(2))
        return self.relu(out + res).unsqueeze(2)  # 增加維度以適應 TCN 的結構

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dilation_sizes, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # [9,9,6,6,3,3,2,2]
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size[i], stride=1, 
                                        dilation=dilation_sizes[i],
                                        padding=((kernel_size[i] - 1) * dilation_sizes[i])//2, dropout=dropout))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ===================FTtransformer=================== #
class FTTransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_numeric, 
        num_categorical, 
        categorical_cardinalities,
        embedding_dim=32, 
        depth=4, 
        heads=8, 
        attn_dropout=0.1, 
        ff_dropout=0.1
    ):
        super().__init__()

        # 數值特徵嵌入（Linear 代替 NEmbedding）
        self.num_embedding = nn.Linear(num_numeric, embedding_dim)

        # 類別特徵嵌入（Embedding 代替 CEmbedding）
        self.cat_embedding = nn.ModuleList([
            nn.Embedding(num_classes, embedding_dim) for num_classes in categorical_cardinalities
        ])

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=embedding_dim * 4,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True
            ),
            num_layers=depth
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, numeric_features, categorical_features):
        batch_size = numeric_features.size(0)

        # 數值特徵嵌入
        num_encoded = self.num_embedding(numeric_features)

        # 類別特徵嵌入
        cat_encoded = [emb(categorical_features[:, i]) for i, emb in enumerate(self.cat_embedding)]
        cat_encoded = torch.stack(cat_encoded, dim=1).mean(dim=1)  # 池化合併

        # 合併數值型與類別型特徵
        x = num_encoded + cat_encoded  # (batch_size, embedding_dim)

        # 加入 CLS Token
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, embedding_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (batch_size, seq_len + 1, embedding_dim)

        # Transformer Encoder
        x = self.transformer(x)  # (batch_size, seq_len + 1, embedding_dim)

        return x[:, 0, :]  # 只取 CLS Token 作為輸出

#===============================================#

class hybrid_model(nn.Module):
    def __init__(
        self, 
        seq_input_dim,
        seq_hidden_dim,
        num_att_head,
        num_tcn_layer,
        tcn_dilation_sizes,
        tcn_kernel_size,
        tcn_dropout,
        num_numeric_features,
        num_categorical_features,
        categorical_cardinalities,
    ):
        super().__init__()
        self.seq_hidden_dim = seq_hidden_dim
        self.seq_input_dim = seq_input_dim
        self.tcn = TemporalConvNet(
            seq_input_dim, 
            num_channels=[seq_hidden_dim] * num_tcn_layer, 
            dilation_sizes=tcn_dilation_sizes, 
            kernel_size=tcn_kernel_size, 
            dropout=tcn_dropout
        )

        # === FT-Transformer ===
        self.ft_transformer = FTTransformerEncoder(
            num_numeric=num_numeric_features,
            num_categorical=num_categorical_features,
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=seq_hidden_dim,
            depth=2,
            heads=4,
            attn_dropout=0.3,
            ff_dropout=0.3
        )

        # 定義類別與數值特徵的索引
        self.categorical_indices = [FEATURES.index(col) for col in CATEGORICAL_FEATURES]
        self.numeric_indices = [FEATURES.index(col) for col in NUMERIC_FEATURES]
        

        self.final_clf = nn.Sequential(
            nn.Linear(seq_hidden_dim*2, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
        )

        self.weight = nn.Sequential(
            nn.Linear(1, seq_hidden_dim*2),
            nn.Sigmoid(),
        )
        


    def forward(self, triage, timeseries, time_steps):
        batch_size = timeseries.size(0)

        # === 這裡拆分 triage ===
        triage_numeric = triage[:, self.numeric_indices].float()  
        triage_categorical = triage[:, self.categorical_indices].long()  

        # ===TCN===
        tcn_input = torch.transpose(timeseries, 1, 2).unsqueeze(2)  
        tcn_features = self.tcn(tcn_input).squeeze(2)  
        tcn_features = tcn_features.mean(dim=2)  

        # === FT-Transformer（靜態特徵）===
        triage_features = self.ft_transformer(triage_numeric, triage_categorical)

        # ===fusion===
        fusion = torch.cat((tcn_features, triage_features), 1)

        # ===weighted sum===
        # 🔑 確保 time_steps 是 float 並 reshape
        time_steps = time_steps.float().view(batch_size, 1)
        weights = self.weight(time_steps)

        logits = self.final_clf(fusion * weights)

        return logits


class MyModel(object):
    def __init__(self):
        print("Initializing")
        import json
        import numpy as np
        # from loss_func import FocalLoss

        # 固定的特徵定義（從外部帶入也可）
        self.CATEGORICAL_FEATURES = ['sex','season','emgdeptchin_name','dayzone','judegmentcode_new',
                                     'maritalstatuscode_new','triage','diagnosisserious','ambulance','nojob']
        self.NUMERIC_FEATURES = ['age','los','weekend','weekend_dischargetime','dayzone_discharge',
                                 'houseincome','sex_vs','vs_age','taiwan']
        self.FEATURES = self.CATEGORICAL_FEATURES + self.NUMERIC_FEATURES
        self.TS_FEATURES = ['dbp', 'sbp', 'rr', 'spo2', 'bt', 'hr']

        # 設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型（請自行調整參數）
        p = {
            "seq_input_dim": 6,
            "seq_hidden_dim": 8,
            "num_att_head": 8,
            "num_tcn_layer": 4,
            "tcn_dilation_sizes": [1, 2, 4, 8],
            "tcn_kernel_size": [3, 3, 3, 2],
            "tcn_dropout": 0.3,
        }

        CAT_DIM = [3, 5, 6, 4, 9, 4, 6, 3, 3, 3]

        self.model = hybrid_model(
            seq_input_dim=p["seq_input_dim"],
            seq_hidden_dim=p["seq_hidden_dim"],
            num_att_head=p["num_att_head"],
            num_tcn_layer=p["num_tcn_layer"],
            tcn_dilation_sizes=p["tcn_dilation_sizes"],
            tcn_kernel_size=p["tcn_kernel_size"],
            tcn_dropout=p["tcn_dropout"],
            num_numeric_features=len(self.NUMERIC_FEATURES),
            num_categorical_features=len(self.CATEGORICAL_FEATURES),
            categorical_cardinalities=CAT_DIM,
        )
        self.model.load_state_dict(torch.load('./model/best_model.ckpt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X, features_names=None):
        print("Predicting")
        print(f"Input type: {type(X)}")

        if isinstance(X, dict):
            try:
                entry = X["data"]["ndarray"][0]
            except Exception as e:
                raise ValueError(f"Dict 格式錯誤: {e}")
        
        elif isinstance(X, np.ndarray):
            try:
                # ndarray -> dict
                entry = X[0]
                if isinstance(entry, bytes):  # 有時候會被轉成 bytes
                    import json
                    entry = json.loads(entry.decode("utf-8"))
                elif not isinstance(entry, dict):  
                    raise ValueError(f"ndarray[0] 不是 dict, 而是 {type(entry)}")
            except Exception as e:
                raise ValueError(f"ndarray 格式錯誤: {e}")
        
        else:
            raise ValueError(f"Unsupported input format: {type(X)}")

        print("Parsed entry:", entry)


        static_data = [float(entry[feat]) for feat in self.FEATURES]
        static_tensor = torch.tensor(static_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        ts_matrix = []
        for row in entry["vital_signs"]:
            ts_matrix.append([
                float(row.get(feat)) if row.get(feat) is not None else 0.0
                for feat in self.TS_FEATURES
            ])
        ts_tensor = torch.tensor(ts_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)

        time_steps_tensor = torch.tensor([[ts_tensor.shape[1]]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(static_tensor, ts_tensor, time_steps_tensor)
            prob = torch.sigmoid(output).item()

        return {
            "data": {
                "names": ["output"],
                "ndarray": [[prob]]
            }
        }


if __name__ == "__main__":
    import json
    with open('./sample_input.json', 'r') as f:
        sample_input = json.load(f)
    model = MyModel()
    result = model.predict(sample_input)    
    print(result)