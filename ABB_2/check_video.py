import cv2


video=r"data2/test/Exp002/Test_Angle2_real.mp4"


cap=cv2.VideoCapture(video)


fps=cap.get(
    cv2.CAP_PROP_FPS
)


frames=cap.get(
    cv2.CAP_PROP_FRAME_COUNT
)


time=frames/fps


print("FPS:",fps)

print("帧数:",frames)

print("视频时间:",time,"秒")