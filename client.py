import socket

tiredness = 1
tiredness_b = tiredness.to_bytes(4, "big")
print(tiredness_b)
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect(("x.x.x.x", 51300))
c.send(tiredness_b)
