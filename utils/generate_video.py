import cv2
import numpy as np


def PSNR(I, O):
    MSE = np.mean((I-O)**2)
    if np.max(I) <= 1:
        I = I * 255
    if np.max(O) <= 1:
        O = O * 255
    return 10 * np.log10(255**2 / MSE)

def addText(images, text, region=(0, 50),font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3., color=(0,255,255), thickness=8):
    if len(images.shape) == 4:
        for i in range(images.shape[0]):
            images[i] = cv2.putText(images[i], text, region, font, fontScale, color, thickness)
    else:
        images = cv2.putText(images, text, (0,100), font, fontScale, color, thickness)
    return images

def PSNR_Videos(filename, ori_path, res_path, frequency=1):
    ori_cap = cv2.VideoCapture(ori_path)
    res_cap = cv2.VideoCapture(res_path)

    ori_fps = ori_cap.get(cv2.CAP_PROP_FPS)
    res_fps = res_cap.get(cv2.CAP_PROP_FPS)
    ori_size = (int(ori_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(ori_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    res_size = (int(res_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(res_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert ori_size == res_size
    ori_fnums = int(ori_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res_fnums = int(res_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert ori_fnums == res_fnums
    print('ori: fps=%3f, size=(%d, %d), FNUMS=%d' %(ori_fps, ori_size[0], ori_size[1], ori_fnums))
    print('res: fps=%3f, size=(%d, %d), FNUMS=%d' %(res_fps, res_size[0], res_size[1], res_fnums))
    success, ori_frame = ori_cap.read()
    assert success
    success, res_frame = res_cap.read()
    assert success
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, ori_fps, (ori_size[0]*2, ori_size[1]))
    while success:
        frame = np.hstack((res_frame, ori_frame))
        img = addText(frame, 'psnr=%5f' % PSNR(res_frame, ori_frame))
        out.write(img)
        success, ori_frame = ori_cap.read()
        success, res_frame = res_cap.read()
    ori_cap.release()
    res_cap.release()
    out.release()
    cv2.destroyAllWindows()
