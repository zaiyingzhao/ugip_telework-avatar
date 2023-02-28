import tkinter as tk
from time import sleep
import threading


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


def load_tiredness():
    f = open("tiredness.txt", "r")
    idx = f.read()
    f.close()
    global index
    if float(idx) < 0.33:
        index = 0
    elif float(idx) < 0.66:
        index = 1
    else:
        index = 2
    return 0


def update_gif():
    global gif_player
    gif_player.stop_loop()
    load_tiredness()
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


if __name__ == "__main__":
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

    avatar_index = 0
    index = 0
    load_tiredness()

    root = tk.Tk()
    root.title("remote-avatar")
    # TODO: gifのサイズをリサイズして統一する
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
