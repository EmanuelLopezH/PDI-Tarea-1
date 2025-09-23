# 01_calibrate_scale.py
import cv2 as cv, numpy as np, sys, os

IMG_FROM_VIDEO = "V2.mp4"
OUT_TXT = "output/scale.txt"
os.makedirs("output", exist_ok=True)

cap = cv.VideoCapture(IMG_FROM_VIDEO)
ok, frame = cap.read(); cap.release()
assert ok, "Cannot read first frame from data/video.mp4"
img = frame.copy()
img = cv.resize(img,(320, 480))

pts=[]
def on_mouse(e,x,y,flags,params):
    if e==cv.EVENT_LBUTTONDOWN and len(pts)<2:
        pts.append((x,y)); cv.circle(img,(x,y),5,(0,255,0),-1); cv.imshow("calibrate",img)

cv.imshow("calibrate", img); cv.setMouseCallback("calibrate", on_mouse)
print("Click two points that are D meters apart (e.g., ruler marks). Press Esc to finish.")
while cv.waitKey(1)!=27: pass
cv.destroyAllWindows()
assert len(pts)==2, "Need exactly 2 clicks"

D_m = float(input("Enter real-world distance in meters between the points: "))
px = np.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1])
px_per_meter = px / D_m
open(OUT_TXT,"w").write(f"{px_per_meter:.6f}")
print("Saved scale at", OUT_TXT, "px_per_meter=", px_per_meter)
