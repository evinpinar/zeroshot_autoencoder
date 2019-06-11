import torch
import torch.nn as nn

class VideoEncoder(torch.nn.Module):

    def __init__(self,
                 C,
                 N,
                 num_feats,
                 drop_out):
        super(VideoEncoder, self).__init__()

        self.C = C
        self.N = N
        self.num_feats = num_feats
        mid = int((N+C)/2)
        
        self.fc = nn.Sequential(
            nn.Linear(num_feats, C),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(C, mid),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(mid, N)
        )     

    def forward(self, inp):
        out = self.fc(inp)
        return out
    
class VideoDecoder(torch.nn.Module):

    def __init__(self,
                 C,
                 N,
                 num_feats,
                 drop_out):
        super(VideoDecoder, self).__init__()

        self.C = C
        self.N = N
        self.num_feats = num_feats
        mid = int((num_feats+C)/2)
        
        self.fc = nn.Sequential(
            nn.Linear(N, C),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(C, mid),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(mid, num_feats)
        )  
        
    def forward(self, inp):
        out = self.fc(inp)
        return out
    
class WordEncoder(torch.nn.Module):

    def __init__(self,
                 L,
                 N,
                 num_feats,
                 drop_out):
        super(WordEncoder, self).__init__()

        self.L = L
        self.N = N
        self.num_feats = num_feats
        mid = int((L+N)/2)
        
        self.fc = nn.Sequential(
            nn.Linear(num_feats, L),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(L, mid),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(mid, N)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out
    
class WordDecoder(torch.nn.Module):

    def __init__(self,
                 L,
                 N,
                 num_feats,
                 drop_out):
        super(WordDecoder, self).__init__()

        self.L = L
        self.N = N
        self.num_feats = num_feats
        mid = int((L+num_feats)/2)
        self.fc = nn.Sequential(
            nn.Linear(N, L),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(L, mid),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(mid, num_feats)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out