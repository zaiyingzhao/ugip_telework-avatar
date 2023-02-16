from PIL import Image, ImageTk
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
    return int(idx)


if __name__ == "__main__":
    # TODO: 動的にgifを変更できるようにする
    paths = ["./gif/piyopiyo.gif", "./gif/loading-hiyoko.gif"]
    idx = load_tiredness()

    root = tk.Tk()
    root.title("remote-avatar")
    # TODO: gifのサイズをリサイズして統一する
    root.geometry("900x700")

    main_frame = tk.Frame(root)
    main_frame.pack()

    label = tk.Label(main_frame)
    label.pack()

    gif_player = TkGif(paths[idx], label)
    gif_player.play()

    root.mainloop()
