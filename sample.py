# -*- coding: utf-8 -*-

# 参考 https://qiita.com/tnoce/items/c819c85a85c16d246be8
import cv2

def main():
    # 入力画像の読み込み
    img = cv2.imread("test.jpg")

    # カスケード型識別器の読み込み
    # https://github.com/opencv/opencv/tree/master/data/haarcascades から
    # 必要なカスケードファイルをダウンロードする

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔領域の探索
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 顔領域を赤色の矩形で囲む
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y+h), (0,0,300), 4)

    # 結果を出力
    cv2.imwrite("result.jpg",img)


if __name__ == '__main__':
    main()
