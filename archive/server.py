import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 51300))
s.listen(1)
while True:
    clientsocket, address = s.accept()
    print("connection established!")
    while True:
        data_b = clientsocket.recv(1024)
        if not data_b:
            break
        data_int = int.from_bytes(data_b, "big")
        print(data_int)
        with open("tiredness.txt", "w") as f:
            f.write(str(data_int))
        f.close()

    clientsocket.close()
s.close()
