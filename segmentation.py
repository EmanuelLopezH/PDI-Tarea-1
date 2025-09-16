import cv2 as cv

video_path = 'V2.mp4'
cap = cv.VideoCapture(video_path)
output_folder = 'output'

# validate if cap is open
if not cap.isOpened():
    print("Error: no se pudo abrir video")
else:
    print("video abierto correctamente")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame o se llegó al final del video")
        break
    cv.imwrite(f'img_{count:04d}.jpg', frame)
    count += 1

cap.release()
cv.destroyAllWindows()
