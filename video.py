import cv2 as cv
import torch
import torch.nn.functional as F

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

c_kernel = torch.tensor([
                        [0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]
                        ], dtype=torch.float32)


kernel = c_kernel.view(1,1,3,3)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("can't receive frame... Exiting")
        break
    # Our operations on the frame come here
    
    print(frame.shape)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_tensor = torch.from_numpy(gray).to(torch.float32)[None, None, :, :]
    
    # Display the resulting frame
    output = F.conv2d(gray_tensor, kernel, padding=1)
    output_frame = cv.convertScaleAbs(output.squeeze().detach().numpy())

    cv.imshow('Input', frame)
    cv.imshow('Output', output_frame)
    #cv.imshow('Output', frame + output_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
