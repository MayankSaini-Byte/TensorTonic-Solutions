import pandas as pd
import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        if not seqs: return np.empty((0, 0))
        max_len = max(len(seq) for seq in seqs)
    
    df = pd.DataFrame(seqs)
    df = df.fillna(pad_value)
    S = df.to_numpy()
    
    current_len = S.shape[1] if S.shape[0] > 0 else 0

    if current_len > max_len:
        Result = S[:, :max_len] 
    elif current_len < max_len:
        pad_width = ((0, 0), (0, max_len - current_len))
        Result = np.pad(S, pad_width, mode='constant', constant_values=pad_value)
    else:
        Result = S
        
    if isinstance(pad_value, int):
        Result = Result.astype(int)

    return Result