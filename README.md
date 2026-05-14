# Neural Style Transfer with Timelapse

VGG19を用いたニューラル画風変換（Neural Style Transfer）の実装です。
コンテンツ画像の「形」を維持しつつ、スタイル画像の「画風」を合成します。
本プロジェクトでは、変換の過程をリアルタイムで確認でき、さらにタイムラプス動画として保存する機能を備えています。

## 特徴

- **高精度な画風変換**: VGG19の学習済みモデルを使用。
- **リアルタイム可視化**: 変換プロセスをOpenCVのウィンドウでリアルタイムに表示。
- **タイムラプス生成**: ステップごとの変化を `mp4` 動画として自動保存。
- **GPU対応**: CUDA環境があれば自動的に高速演算モード（High Performance Mode）で動作。
- **輝度補正機能**: スタイル画像が暗い場合でも、自動的にヒストグラム平坦化を行い最適化します。

## フォルダ構成

```text
Neural-Style-Transfer/
├── images/                # 入力画像の保存先
│   ├── content.jpg        # 変換したい元の写真
│   └── style.jpg          # 適用したい画風の画像
├── outputs/               # 生成された画像・動画の出力先
├── requirements.txt       # 必要ライブラリ一覧
└── main.py                # メインプログラム
```

## セットアップ

### 1. ライブラリのインストール

```bash
pip install -r requirements.txt
```

※ GPU (CUDA) を使用する場合は、PyTorch公式サイト を参照して、環境に合った `torch` をインストールしてください。

### 2. 画像の準備

`images/` フォルダの中に、以下の名前で画像を配置してください。

- `content.jpg` (形を維持したい画像)
- `style.jpg` (画風を借りたい画像)

## 使い方

メインスクリプトを実行します。

```bash
python main.py
```

実行が完了すると、`outputs/` フォルダに以下のファイルが生成されます。

- `result_final.jpg`: 完成した画像
- `process_timelapse.mp4`: 変換プロセスの動画

## パラメータの調整

`main.py` 内の以下の変数を書き換えることで、結果を微調整できます。

- `style_weight`: 画風をどのくらい強く反映するか（デフォルト: 1,000,000）
- `content_weight`: 元の形をどのくらい維持するか（デフォルト: 1）
- `num_steps`: 計算回数（多いほど詳細になりますが時間がかかります）
- `imsize`: 処理する画像サイズ（GPUメモリが足りない場合は下げてください）

## requirements.txt

`pip install -r requirements.txt` で一括インストールするためのファイルです。

```text
torch
torchvision
numpy
Pillow
opencv-python