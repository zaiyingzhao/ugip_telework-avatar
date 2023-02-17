import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START = 1.0
END = 0.0
BETA = 0.5

def tiredness_linear(length, start, end): #start, end 始まりと終わりの疲れ具合(0~1)
    return np.linspace(start, end, length)
def tiredness_sigmoid(length, start, end, beta, alpha=8): #beta 作業時間のうちどのくらいで疲れを感じ始めたか(0~1)
    x = np.linspace(-1,1,length)
    beta = beta*2-1
    y = 1/(1+np.exp(-alpha*(-x-beta))) * (start-end) +end
    return y

if __name__ == "__main__":
    df = pd.read_csv("res.csv")
    length = len(df)
    #df["tiredness"] = tiredness_linear(length,START,END)
    df["tiredness"] = tiredness_sigmoid(length,START,END,BETA)
    df.to_csv("res_tiredness.csv")