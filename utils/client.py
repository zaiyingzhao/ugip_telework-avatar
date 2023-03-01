import socket

def connect2server():
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect(("157.82.200.228", 51300))

    return c

def send2server(c, tiredness):
    tiredness_b = tiredness.to_bytes(4, "big")
    c.send(tiredness_b)

