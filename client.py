import socket

target_host = ""
target_port = 50000
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((target_host, target_port))
while True:
    while True:
        print("waiting for his response...")
        rec = client.recv(1024)
        # fd = open("tiredness.txt", "wb")
        # fd.write(rec)
        # fd.close()
        print(rec)
        if len(rec) == 0:
            client.close()
            break
    # while True:
    #     file = input("textfile: ")
    #     try:
    #         fd = open(file, "rb")
    #     except FileNotFoundError:
    #         print("can't open this file")
    #         print("please try it again")
    #         print("")
    #     else:
    #         text = fd.read()
    #         fd.close()
    #         break
    # client.send(text)
    # print("succeed")
    # print("")
