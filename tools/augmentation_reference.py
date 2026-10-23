#!/usr/bin/env python3
"""Regenerate fixed u8 C++/OpenCV augmentation fixtures (requires cv2/numpy).
No training; reproduces image_opencv.cpp's crop/mean/resize/flip/HSV path.
"""
from pathlib import Path
import cv2
import numpy as np

src = np.array([(i * 37 + 11) % 256 for i in range(7 * 5 * 3)], dtype=np.uint8).reshape(5, 7, 3)
rows = ["# left,top,crop_w,crop_h,flip,out_w,out_h,hue,sat,exp,rgb_hex"]
for left, top, cw, ch, flip, nw, nh in [(0,0,7,5,0,7,5), (1,1,5,3,1,9,4), (-2,-1,10,8,0,6,5), (2,0,4,5,0,3,9), (0,2,7,2,1,8,6)]:
    for hue, sat, exp in [(0,1,1), (.1,1.5,.8), (-.1,.7,1.3), (.02,1.1,1.1)]:
        crop = np.empty((ch,cw,3), dtype=np.uint8)
        crop[:] = np.rint(src.mean(axis=(0,1))).astype(np.uint8)
        for y in range(ch):
            for x in range(cw):
                if 0 <= x+left < 7 and 0 <= y+top < 5:
                    crop[y,x] = src[y+top,x+left]
        out = cv2.resize(crop, (nw,nh), interpolation=cv2.INTER_LINEAR)
        if flip:
            out = cv2.flip(out, 1)
        if hue or sat != 1 or exp != 1:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
            # Mat *= / += use saturate_cast<uchar>(cvRound(value)).
            hsv[:,:,1] = np.clip(np.rint(hsv[:,:,1].astype(np.float32)*sat),0,255).astype(np.uint8)
            hsv[:,:,2] = np.clip(np.rint(hsv[:,:,2].astype(np.float32)*exp),0,255).astype(np.uint8)
            hsv[:,:,0] = np.clip(np.rint(hsv[:,:,0].astype(np.float32)+179*hue),0,255).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rows.append(','.join(map(str,[left,top,cw,ch,flip,nw,nh,hue,sat,exp]))+','+out.tobytes().hex())
path = Path(__file__).resolve().parents[1] / 'src/lightnet-core/tests/data/augmentation_opencv.csv'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('\n'.join(rows)+'\n')
print(path)
