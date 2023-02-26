import socket

bind_host = ""
bind_port = 50000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_host, bind_port))
server.listen(5)
print("host: " + bind_host)
print("port: " + str(bind_port))

tiredness = 1

while True:
    client, addr = server.accept()
    # print("from:" + addr[0] + " " + str(addr[1]))
    # fd = open("tiredness.txt", "rb")
    # text = fd.read()
    # fd.close()
    server.send(tiredness)
    # while True:
    # print("waiting for his response...")
    # rec = client.recv(1024)
    # fd = open("tiredness.txt", "wb")
    # fd.write(rec)
    # fd.close()
    # print(rec.decode("utf-8"))
    # if len(rec) == 0:
    #     client.close()
    #     break
