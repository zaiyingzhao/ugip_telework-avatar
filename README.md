# ugip_telework-avatar

## セットアップ
1. face++のAPIを取得する
2. `utils/api.py`中の`API_KEY`, `API_SECRET`を1.で取得した値に設定する。
3. `utils/client.py`の`IP`,`PORT`を疲労度を送りたい相手のIPアドレス、ポート番号に設定する。(おそらく2人で同じwifiに接続していないと動きません)

## 使い方
1. 個人にフィットしたモデルを作る
  ```bash
  $ python setup_main.py
  ```
  - 上記コマンドを実行して、1日働いてください。
  - 実行すると
    ```bash
    tell me how tired you are now(0~1):
    ```
    と聞かれるので、仕事前の疲労度を0~1の小数で入力してください。入力が完了すると、データの収集を始めます。
  - 働いている間は定期的に写真をとり、face++で得られたデータを蓄積していきます。
  - 仕事が終わったら、カメラ画像が表示されているウィンドウにフォーカスして、"q"のキーを押してください。これが仕事終了の合図です。
  - すると再び
    ```bash
    tell me how tired you are now(0~1):
    ```
    と聞かれるので、仕事後の疲労度を回答してください。
  - 次に
    ```bash
    tell me when you started to feel tired(0~1): 
    ```
    と聞かれるので、どのくらいのタイミングで疲れを感じ始めたか入力してください。例えば9時から12時まで働いて、11時ごろに疲れを感じ始めたら、0.67と入力します。
  - 取得した顔データと、入力内容に基づいてあなたの「疲れ」の変化を学習します。

2. 実際に疲れを推定する(2人で行う)
  - 一人は
    ```bash
    $ python calc_tiredness.py
    ```
    を実行してください。先ほど作成したモデルに基づいて現在の疲労度を相手に送ります。
  - もう一人は
    ```bash
    python 
    ```
    を実行してください。相手から送られてきた疲労度を元に相手のアバターを変化させます。
<!-- 3. transfer data(less important)
  - socket(TCP/IP) communication  


4. visualize tiredness/current working condition
  - sticky notes(windows)
  - tkinter(python module)
  - modify desktop goose(if possible) -->
