import pandas as pd
import numpy as np

from utils import collect_data

def tiredness_linear(length, start, end): #start, end 始まりと終わりの疲れ具合(0~1)
    return np.linspace(start, end, length)
def tiredness_sigmoid(length, start, end, beta, alpha=8): #beta 作業時間のうちどのくらいで疲れを感じ始めたか(0~1)
    alpha=20
    x = np.linspace(-1,1,length)
    beta = beta*2-1
    y = 1/(1+np.exp(-alpha*(x-beta))) * (end-start) + start
    return y

def add_tiredness(START:float = 1.0, END:float = 0.0, BETA:float = 0.5):
    df = pd.read_csv(f"{collect_data.OUTPUT_FOLDER}/res.csv", index_col=0)
    length = len(df)
    df["tiredness"] = tiredness_sigmoid(length,START,END,BETA)
    df.to_csv(f"{collect_data.OUTPUT_FOLDER}/res_tiredness.csv")