import time

if __name__ == "__main__":
    while True:
        f = open("tiredness.txt", "w")
        f.write("1")
        f.close()
        time.sleep(5)
        f = open("tiredness.txt", "w")
        f.write("0")
        f.close()
        time.sleep(5)
