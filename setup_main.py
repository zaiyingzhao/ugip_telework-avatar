from utils import collect_data
from utils import add_label
from utils import standardize
from utils import model

if __name__ == "__main__":
    start = input("tell me how tired you are now(0~1): ")
    cap = collect_data.set_cap()
    collect_data.collect_data(cap)
    cap.release()
    
    end = input("tell me how tired you are now(0~1): ")
    beta = input("tell me when you started to feel tired(0~1): ")
    add_label.add_tiredness(START=float(start), END=float(end), BETA=float(beta))
    standardize.standardize_data()
    model.make_model()