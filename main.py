import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os
import copy

# ==========================================
# 1. 設定エリア
# ==========================================
# CUDA (GPU) が使えるか確認してデバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# デバイス情報を表示
print("========================================")
if device.type == 'cuda':
  print(
      f"🚀 High Performance Mode: CUDA (GPU) を使用します: {torch.cuda.get_device_name(0)}")
  torch.backends.cudnn.benchmark = True  # 高速化

  # メモリ不足対策: 800で落ちる場合は 512 に下げてください
  imsize = 800
else:
  print(f"⚠️ Standard Mode: CPU を使用します")
  imsize = 512

print(f"処理解像度: {imsize} x {imsize} (最大長辺)")
print("========================================")

# ファイルパス設定
content_path = "tatsumi.jpg"     # 顔写真
style_path = "images.jpg"    # 画風画像
output_video_path = "process_timelapse.mp4"  # 完成過程の動画

# 画風適用の強さ調整
style_weight = 1000000  # 画風の強さ
content_weight = 1      # 元の形の強さ
num_steps = 200        # 計算回数

# ==========================================
# 2. 画像の読み込みと前処理
# ==========================================
def image_loader(image_name, is_style=False):
  if not os.path.exists(image_name):
    print(f"❌ エラー: ファイル '{image_name}' が見つかりません。")
    exit()

  image = Image.open(image_name)

  # スタイル画像（モナリザ）が暗い場合の補正処理 (OpenCV使用)
  if is_style:
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 輝度ヒストグラム平坦化
    img_yuv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    cv_img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image = Image.fromarray(cv2.cvtColor(cv_img_eq, cv2.COLOR_BGR2RGB))

  # リサイズとTensor変換
  loader = transforms.Compose([
      transforms.Resize(imsize),
      transforms.CenterCrop(imsize),
      transforms.ToTensor()
  ])

  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

# 画像をロード
print("画像を読み込んでいます...")
style_img = image_loader(style_path, is_style=True)
content_img = image_loader(content_path, is_style=False)

# ==========================================
# 3. Loss関数とモデル定義
# ==========================================
def gram_matrix(input):
  a, b, c, d = input.size()
  features = input.view(a * b, c * d)
  G = torch.mm(features, features.t())
  return G.div(a * b * c * d)

class ContentLoss(nn.Module):
  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    self.loss = F.mse_loss(input, self.target)
    return input

class StyleLoss(nn.Module):
  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    # ターゲットの特徴量をグラム行列に変換して保存
    self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
    gram_input = gram_matrix(input)
    self.loss = F.mse_loss(gram_input, self.target)
    return input

class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    self.std = torch.tensor(std).view(-1, 1, 1).to(device)

  def forward(self, img):
    return (img - self.mean) / self.std

# VGG19モデルのロード
print("VGG19モデルを構築中...")
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
  normalization = Normalization(
      normalization_mean, normalization_std).to(device)
  content_losses = []
  style_losses = []

  model = nn.Sequential(normalization)

  content_layers = ['conv_4']
  style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

  i = 0
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
      name = 'relu_{}'.format(i)
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
      name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
      name = 'bn_{}'.format(i)
    else:
      raise RuntimeError('Unrecognized layer: {}'.format(
          layer.__class__.__name__))

    model.add_module(name, layer)

    if name in content_layers:
      target = model(content_img).detach()
      content_loss = ContentLoss(target)
      model.add_module("content_loss_{}".format(i), content_loss)
      content_losses.append(content_loss)

    if name in style_layers:
      target_feature = model(style_img).detach()
      style_loss = StyleLoss(target_feature)
      model.add_module("style_loss_{}".format(i), style_loss)
      style_losses.append(style_loss)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
      break
  model = model[:(i + 1)]

  return model, style_losses, content_losses

# ==========================================
# 4. 学習ループ実行
# ==========================================
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=0.5):

  print("モデルセットアップ完了。変換プロセスを開始します...")
  model, style_losses, content_losses = get_style_model_and_losses(
      cnn, normalization_mean, normalization_std, style_img, content_img)

  # 【重要修正】入力画像を更新可能(requires_grad=True)にする
  input_img.requires_grad_(True)
  # モデルのパラメータは固定する
  model.requires_grad_(False)

  # オプティマイザ設定
  optimizer = optim.LBFGS([input_img])

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  _, _, h, w = input_img.shape
  video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))

  print(f"ウィンドウ表示と動画保存('{output_video_path}')を開始します。")

  run = [0]
  while run[0] <= num_steps:

    def closure():
      input_img.data.clamp_(0, 1)

      optimizer.zero_grad()
      model(input_img)

      style_score = 0
      content_score = 0

      for sl in style_losses:
        style_score += sl.loss
      for cl in content_losses:
        content_score += cl.loss

      style_score *= style_weight
      content_score *= content_weight

      loss = style_score + content_score
      loss.backward()

      run[0] += 1
      if run[0] % 50 == 0:
        print("Step {}: Style Loss : {:4f} Content Loss: {:4f}".format(
            run[0], style_score.item(), content_score.item()))

      return style_score + content_score

    optimizer.step(closure)

    # 可視化
    if run[0] % 5 == 0:
      image = input_img.clone().detach()
      image.data.clamp_(0, 1)
      image = image.cpu().squeeze(0)

      image_pil = transforms.ToPILImage()(image)
      img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

      cv2.putText(img_cv, f"Step: {run[0]}", (10, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

      cv2.imshow("Real-time Process", img_cv)
      cv2.waitKey(1)
      video_writer.write(img_cv)

  video_writer.release()
  cv2.destroyAllWindows()

  image = input_img.clone().detach()
  image.data.clamp_(0, 1)
  image = image.cpu().squeeze(0)
  final_image = transforms.ToPILImage()(image)
  final_image.save("result_final.jpg")
  print("✅ 全工程完了！ 'result_final.jpg' を保存しました。")

  return input_img

# ==========================================
# メイン実行
# ==========================================
input_img = content_img.clone()

try:
  output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                              content_img, style_img, input_img,
                              num_steps=num_steps,
                              style_weight=style_weight,
                              content_weight=content_weight)
except Exception as e:
  print("\n❌ エラーが発生しました:")
  import traceback
  traceback.print_exc()
