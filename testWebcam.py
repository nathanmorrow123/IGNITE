import cv2

capture = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
capture.set(3, frameWidth)
capture.set(4, frameHeight)

cv2.waitKey(1000)
_, referenceimg = capture.read()

while True:

    _, current_frame = capture.read()
    # Change images to grayscale
    reference_img_bw = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    current_frame_bw = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # ORB object
    orb = cv2.ORB_create()

    refKeypoints, refDescriptors = orb.detectAndCompute(reference_img_bw, None)
    frameKeypoints, frameDescriptors = orb.detectAndCompute(current_frame_bw, None)

    print(str(frameKeypoints[0]) + ' ' + str(frameDescriptors[0]))

    matcher = cv2.BFMatcher()
    matches = matcher.match(refDescriptors, frameDescriptors)

    final_img = cv2.drawMatches(reference_img, refKeypoints,
                                current_frame, frameKeypoints, matches[:1], None)

    final_img = cv2.resize(final_img, (1000, 650))

    cv2.imshow("Matches", final_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()