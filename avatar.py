import tkinter as tk
from time import sleep
import threading
import socket


class GifPlayer(threading.Thread):
    def __init__(self, path: str, label: tk.Label):
        super().__init__(daemon=True)
        self._please_stop = False
        self.path = path
        self.label = label
        self.duration = []
        self.frames = []
        self.last_frame_index = None
        self.load_frames()

    def load_frames(self):
        frames = []
        frame_index = 0
        try:
            while True:
                frames.append(
                    tk.PhotoImage(file=self.path, format=f"gif -index {frame_index}")
                )
                frame_index += 1
        except Exception:
            self.frames = frames
            self.last_frame_index = frame_index - 1

    def run(self):
        frame_index = 0
        while not self._please_stop:
            self.label.configure(image=self.frames[frame_index])
            frame_index += 1
            if frame_index > self.last_frame_index:
                frame_index = 0
            sleep(0.3)

    def stop(self):
        self._please_stop = True


class TkGif:
    def __init__(self, path, label: tk.Label) -> None:
        self.path = path
        self.label = label

    def play(self):
        self.player = GifPlayer(self.path, self.label)
        self.player.start()

    def stop_loop(self):
        self.player.stop()


def update_gif():
    global gif_player
    gif_player.stop_loop()
    gif_player = TkGif(paths[avatar_index][index], label)
    gif_player.play()


# execute update_gif() every 1 sec.
def repeat_func():
    update_gif()
    root.after(1000, repeat_func)


# if button is pressed, increase avatar_index by 1
def change_avatar():
    n = len(paths)
    global avatar_index
    avatar_index = (avatar_index + 1) % n


def server():
    global s
    global index
    global valid
    while True:
        clientsocket, address = s.accept()
        print("connection established!")
        while True:
            data_b = clientsocket.recv(1024)
            if not data_b:
                break
            index = int.from_bytes(data_b, "big")

        clientsocket.close()


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    PORT = 51300
    s.bind(("0.0.0.0", PORT))
    s.listen(1)

    avatar_index = 0
    index = 0
    valid = 1

    thread_server = threading.Thread(target=server)
    thread_server.setDaemon(True)
    thread_server.start()

    paths = [
        [
            "./gif/norinoriflower.gif",
            "./gif/piyopiyo.gif",
            "./gif/loading-hiyoko.gif",
        ],
        [
            "./gif/odorupen.gif",
            "./gif/tobipen.gif",
            "./gif/mimimimi.gif",
        ],
    ]

    root = tk.Tk()
    root.title("remote-avatar")
    root.geometry("900x700")

    main_frame = tk.Frame(root)
    main_frame.pack()

    button = tk.Button(
        main_frame,
        text="change avatar",
        font=("MSゴシック", "11", "bold"),
        width=15,
        justify=tk.LEFT,
        command=change_avatar,
    )
    button.pack()

    label = tk.Label(main_frame)
    label.pack()

    gif_player = TkGif(paths[avatar_index][index], label)
    gif_player.play()

    repeat_func()
    root.mainloop()
