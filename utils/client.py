import socket

IP = "YOUR IP"
PORT = 51300

def connect2server():
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((IP, PORT))

    return c

def send2server(c, tiredness):
    tiredness_b = tiredness.to_bytes(4, "big")
    c.send(tiredness_b)

