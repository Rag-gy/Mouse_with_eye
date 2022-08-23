import cv2
import mediapipe as mp
import pyautogui as gui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)
screen_w, screen_h = gui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape
    face_points = output.multi_face_landmarks
    
    if face_points:
        points = face_points[0].landmark

        for id, point in enumerate(points[474:478]):
            x = int(point.x*frame_w)
            y = int(point.y*frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

            if id == 1:
                screen_x = int(point.x*screen_w)
                screen_y = int(point.y*screen_h)
                gui.moveTo(screen_x, screen_y)

        left = [points[145], points[159]]

        for point in left:
            x = int(point.x*frame_w)
            y = int(point.y*frame_h)
            cv2.circle(frame, (x, y), 3, (255, 255, 255))
        if left[0].y-left[1].y < 0.011:
            gui.click()
            gui.sleep(0.9)

    cv2.imshow("Eye controlled Mouse", frame)
    cv2.waitKey(1)