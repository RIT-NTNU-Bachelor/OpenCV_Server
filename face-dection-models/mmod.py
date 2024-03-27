import cv2


def detectFaceDlibMMOD(detector, frame, inHeight=300, inWidth=0, detectMultipleFaces=False):
    frameDlibMMOD = frame.copy()
    frameHeight = frameDlibMMOD.shape[0]
    frameWidth = frameDlibMMOD.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))
    frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibMMODSmall, 0)

    bboxes = []
    for faceRect in faceRects:
        x1 = int(faceRect.rect.left() * scaleWidth)
        y1 = int(faceRect.rect.top() * scaleHeight)
        x2 = int(faceRect.rect.right() * scaleWidth)
        y2 = int(faceRect.rect.bottom() * scaleHeight)
        width = x2 - x1
        height = y2 - y1
        bboxes.append((x1, y1, width, height))

    if detectMultipleFaces == True:
        return bboxes  # Return all detected faces
    else:
        return bboxes[0] if bboxes else None  # Return the first face or None if no faces are detected
