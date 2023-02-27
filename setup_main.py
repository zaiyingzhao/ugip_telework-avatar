from setup import collect_data
from setup import add_label
from setup import standardize
from setup import model

if __name__ == "__main__":
    cap = collect_data.set_cap()
    collect_data.collect_data(cap)
    cap.release()
    add_label.add_tiredness()
    standardize.standardize_data()
    model.make_model()