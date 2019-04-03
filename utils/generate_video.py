import cv2
import numpy as np
import math


def psnr(img1,img2):
	mse=np.mean((img1/255.-img2/255.)**2)
	if mse<1e-10:
		return 100
	return 20 * math.log10(1/math.sqrt(mse))


def generate_video(videoname1,videoname2):
	VC_n=cv2.VideoCapture(videoname1)
	VC_g=cv2.VideoCapture(videoname2)

	fps = VC_n.get(cv2.CAP_PROP_FPS)
	size = (int(VC_n.get(cv2.CAP_PROP_FRAME_WIDTH))*2,
			int(VC_n.get(cv2.CAP_PROP_FRAME_HEIGHT)))


	videoWriter = cv2.VideoWriter('results.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
	success_n,frame_n=VC_n.read()
	success_g,frame_g=VC_g.read()
	count=0
	color=(255,255,255)
	pos=(150,150)
	text_size=2
	font = cv2.FONT_HERSHEY_SIMPLEX
	while success_g and success_n:

		# w,h=size
		# frame=np.zeros((h,w,3),np.uint8)
		# frame[:,0:int(w/2),:]=frame_n[:,:,:]
		# frame[:,int(w/2):w,:]=frame_g[:,:,:]
		frame=np.hstack((frame_n,frame_g))
		if count%30==0:
			txt='psnr:{:.4}'.format(psnr(frame_n,frame_g))
		cv2.putText(frame,txt,pos,font,text_size,color,2)
		count+=1
		videoWriter.write(frame)
		success_n, frame_n = VC_n.read()
		success_g, frame_g = VC_g.read()
	VC_g.release()
	VC_n.release()
	videoWriter.release()

if __name__=='__main__':
	generate_video('../dev-23.mp4','../dev-origin-23.mp4')
