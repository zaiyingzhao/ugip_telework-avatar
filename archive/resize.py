from PIL import Image
import glob

gifs = glob.glob("./gif/*.gif")

for gif in gifs:
    # アニメーションGIFを読み込む
    image = Image.open(gif)

    # 幅と高さを取得
    width, height = image.size

    # アニメーションの各フレームをリサイズしてリストに格納
    resize_image_list = []
    for index in range(image.n_frames):
        image.seek(index)
        resize_image_list.append(image.resize((900, 700)))

    # アニメーションGIFとして書き出し
    resize_image_list[0].save(
        gif,
        save_all=True,
        append_images=resize_image_list[1:],
        loop=0,
    )
