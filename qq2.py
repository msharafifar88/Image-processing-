# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import array, uint8
import pprint
import os


def PlotImg(obj, desc):
	plt.figure(desc)
	plt.imshow(obj, cmap='gray', interpolation='bicubic')
	return True


def BlurImage(img, repetition):
	for i in range(repetition):
		img = cv2.GaussianBlur(img, (5, 5), 0)
	return img


class Image:
	"""Custom image class"""

	def __init__(self, name, width, height, th1, th2):
		self.ImgName = name
		self.OriginalImg = cv2.imread(os.path.join('images', self.ImgName))
		self.ImgResized = self.ReadImage(self.ImgName, width, height)
		self.ImgGray = cv2.cvtColor(self.ImgResized, cv2.COLOR_BGR2GRAY)
		self.ImgGrayBlurred = BlurImage(self.ImgGray, 2)
		self.ImgGrayBlurredEdged = cv2.Canny(self.ImgGrayBlurred, th1, th2)
		# ---- blob detector params ----
		# thresholds
		self.minThreshold = 10
		self.maxThreshold = 200
		# filter by Area
		self.filterByArea = True
		self.minArea = 30
		# Filter by Circularity
		self.filterByCircularity = True
		# self.minCircularity = 0.3
		self.minCircularity = 0.8
		# Filter by Convexity
		self.filterByConvexity = True
		self.minConvexity = 0.87
		# Filter by Inertia
		self.filterByInertia = True
		self.minInertiaRatio = 0.14

	def ReadImage(self, name, width, height):
		img = cv2.imread(os.path.join('images', name))
		img = cv2.resize(img, (width, height))
		return img

	def FindContours(self, img, upto, approxedges):

		# Gaussian Blur to radius false detection
		img = cv2.GaussianBlur(self.ImgGray, (9, 19), 0)
		# Apply Circle hough transform with radius between 200,600 (because images size is 2000x2000 and the dish size is about 50% of image)
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 100, param1=128, minRadius=200, maxRadius=600)
		# draw detected circles on image
		circles = circles.tolist()
		for cir in circles:
			for x, y, r in cir:
				x, y, r = int(x), int(y), int(r)
				height, width = img.shape
				circle_img = np.zeros((height,width), np.uint8)
				cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
				img = cv2.bitwise_and(self.ImgGray, self.ImgGray, mask=circle_img)


		# show the output image
		# cv2.imshow("output", cv2.resize(img, (500, 500)))
		# cv2.waitKey(0)
		# threshold ignore dark a part of image
		ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
		_, cnts, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:upto]
		screenCnt = None
		for c in cnts:
			approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
			if len(approx) > approxedges:
				print 'circle'
				screenCnt = c
				break

		return screenCnt

	def MaskImg(self, img, cnt, r, g, b, thick):
		stamp = False
		imgcopy = img.copy()
		mask = np.zeros_like(imgcopy)

		cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
		if stamp:
			PlotImg(mask, 'MASK')
		kernel = np.ones((20, 20), np.uint8)
		masker = cv2.erode(mask, kernel, iterations=1)
		if stamp:
			PlotImg(masker, 'MASKERODED')
		out = np.zeros_like(imgcopy)
		out[masker == (255, 255, 255)] = imgcopy[masker == (255, 255, 255)]
		cv2.drawContours(imgcopy, [cnt], -1, (r, g, b), thick)
		return imgcopy, out

	def ExtractRoi(self, img, cnt):
		x, y, width, height = cv2.boundingRect(cnt)
		roi = img[y:y + height, x:x + width]
		roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		return roi, roigray

	def IncreaseContrast(self, img, maxIntensity, alfa, beta, scaleFact):
		imgcopy = (maxIntensity / alfa) * (img / (maxIntensity / beta)) ** scaleFact
		imgcopy = array(imgcopy, dtype=uint8)
		return imgcopy

	def ThresholdAndSmooth(self, img, maxIntensity, phi, theta, scaleFactor):
		imgcopy = BlurImage(img, 2)
		ret3, imgthr = cv2.threshold(imgcopy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		print ret3
		imgthr = BlurImage(imgthr, 2)
		imgthr = self.IncreaseContrast(imgthr, maxIntensity, phi, theta, scaleFactor)
		return imgthr

	def ThresholdAdaptive(self, img, maxIntensity, phi, theta, scaleFactor):
		imgcopy = BlurImage(img, 2)
		imgthr = cv2.adaptiveThreshold(imgcopy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		# imgthr = BlurImage(imgthr, 2)
		# imgthr = self.IncreaseContrast(imgthr, maxIntensity, phi, theta, scaleFactor)
		return imgthr

	def ThresholdNormal(self, img, maxIntensity, phi, theta, scaleFactor):
		# imgcopy = BlurImage(img, 2)
		ret, imgthr = cv2.threshold(img.copy(), 2, 255, cv2.THRESH_BINARY)
		ret2, imgthr2 = cv2.threshold(img.copy(), 2, 255, cv2.THRESH_BINARY_INV)
		# imgthr = BlurImage(imgthr, 10)
		# imgthr = self.IncreaseContrast(imgthr, maxIntensity, phi, theta, scaleFactor)
		return imgthr, imgthr2

	def CreateDetector(self):
		params = cv2.SimpleBlobDetector_Params()

		# thresholds
		params.minThreshold = self.minThreshold
		params.maxThreshold = self.maxThreshold

		# filter by Area
		params.filterByArea = self.filterByArea
		params.minArea = self.minArea

		# Filter by Circularity
		params.filterByCircularity = self.filterByCircularity
		params.minCircularity = self.minCircularity

		# Filter by Convexity
		params.filterByConvexity = self.filterByConvexity
		params.minConvexity = self.minConvexity

		# Filter by Inertia
		params.filterByInertia = self.filterByInertia
		params.minInertiaRatio = self.minInertiaRatio

		ver = (cv2.__version__).split('.')
		if int(ver[0]) < 3:
			detector = cv2.SimpleBlobDetector(params)
		else:
			detector = cv2.SimpleBlobDetector_create(params)
		return detector


if __name__ == '__main__':

	"""Global variables"""

	diff = False  # if True, Plot differences of the images
	diffthr = False  # if True, Plot thresholded differences of images
	kpts = False  # if True, Plot images with detected keypoints
	roigray = False  # if True, Plot the grayscale ROI extracted from images
	back = False  # if True, Plot images minus the test20 image used as background
	final_inv = False
	final_inv_blur = False
	final_diff = False
	final_diff_blur = False
	# key = False

	# diff = True
	# diffthr = True
	# kpts = True
	# roigray = True
	# back = True
	# final_inv = True
	# final_diff = True
	# final_inv_blur = True
	# final_diff_blur = True
	key = True

	maxIntensity = 255.0  # depends on dtype of image data

	phi = 2
	theta = 2.9
	scaleFactor = 0.5

	phi2 = 1.1
	theta2 = 1
	scaleFactor2 = 2

	images_names = [f for f in os.listdir("images") if f.endswith('.jpg')]
	images = {}

	masked_images = {}
	out_images = {}
	roi_images = {}

	backsub_images = {}  # tutte le immagini a cui è stato sottratto il background ----> back = True per plottare
	final_bin_images = {}  # threshold BINARIA sulle immagini a cui è stato sottratto il background
	final_inv_images = {}  # threshold BINARIA INVERSA sulle immagini a cui è stato sottratto il background ----> final_inv = True per plottare
	final_inv_images_blurred = {}
	final_diff_images = {}  # risultato della sottrazione a due a due degli elementi di final_bin_images{}
	final_diff_bin_images = {}  # threshold BINARIA degli elementi di final_diff_images{}
	final_diff_inv_images = {}  # threshold BINARIA INVERSA degli elementi di final_diff_images{} ----> final_diff = True per plottare
	final_diff_inv_images_blurred = {}

	roigray_images = {}
	diff_images = {}
	diff_thresh_images = {}
	keypoints_images = {}
	imwhitkeypoints_images = {}

	# ---- INSTANTIATE THE IMAGES ----

	"""Instantiate images"""

	for i, j in zip(images_names, range(len(images_names))):
		images[j] = Image(i, 2000, 2000, 100, 200)
		print ('%s : %s' % (j, images[j].ImgName))

	# ---- FIND THE CENTRAL CIRCLE ----

	"""Find the central circle in the first image and then use it to crop all the images.
	This is possible because the circle is always in the same position in all the images"""

	circle = images[0].FindContours(images[0].ImgGrayBlurredEdged.copy(), 10,
									15)  # the third parameter is used to approx the circle shape
	# ---- EXTRACT ROI FROM IMAGES, THRESHOLD, DRAW KPTS, PLOT ----

	"""Build the mask baased on the bounding rectangles of the central circle detected before"""

	detector = images[0].CreateDetector()

	for i in range(len(images_names)):
		masked_images[i], out_images[i] = images[0].MaskImg(images[i].ImgResized, circle, 0, 255, 0, 3)
		roi_images[i], roigray_images[i] = images[0].ExtractRoi(out_images[i], circle)
		if i == 0:
			diff_images[i] = roigray_images[i]
		else:
			diff_images[i] = cv2.absdiff(roigray_images[i], roigray_images[i - 1])
			diff_images[i] = images[0].IncreaseContrast(diff_images[i], maxIntensity, phi, theta, scaleFactor)

			if roigray:
				PlotImg(roigray_images[i], '%s ROIGRAY' % images[i].ImgName)

			if diff:
				PlotImg(diff_images[i], '%s - %s' % (images[i].ImgName, images[i - 1].ImgName))

			if diffthr:
				diff_thresh_images[i] = images[0].ThresholdNormal(diff_images[i], maxIntensity, phi2, theta2,
																  scaleFactor2)
				PlotImg(diff_thresh_images[i], '%sTHR - %sTHR' % (images[i].ImgName, images[i - 1].ImgName))

			if kpts:
				keypoints_images[i] = detector.detect(diff_thresh_images[i])
				imwhitkeypoints_images[i] = cv2.drawKeypoints(diff_thresh_images[i], keypoints_images[i], np.array([]),
															  (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
				PlotImg(imwhitkeypoints_images[i], 'Immagine %s - %s' % (images[i].ImgName, images[i - 1].ImgName))

	background = roigray_images[2]
	for i in range(len(images_names)):
		backsub_images[i] = cv2.absdiff(roigray_images[i], background)
		backsub_images[i] = BlurImage(backsub_images[i], 50)
		final_bin_images[i], final_inv_images[i] = images[0].ThresholdNormal(backsub_images[i], maxIntensity, phi2,
																			 theta2, scaleFactor2)
		final_inv_images_blurred[i] = BlurImage(final_inv_images[i], 2)
		keypoints = detector.detect(final_inv_images_blurred[i])
		# imwhitkeypoints_images[i] = cv2.drawKeypoints(final_inv_images_blurred[i], keypoints, np.array([]),
		# (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		imwhitkeypoints_images[i] = cv2.drawKeypoints(roigray_images[i], keypoints, np.array([]),
													  (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		if i == 0:
			final_diff_images[i] = final_bin_images[i]

		else:
			final_diff_images[i] = cv2.absdiff(final_bin_images[i], final_bin_images[i - 1])

		final_diff_bin_images[i], final_diff_inv_images[i] = images[0].ThresholdNormal(final_diff_images[i],
																					   maxIntensity, phi2, theta2,
																					   scaleFactor2)
		final_diff_inv_images_blurred[i] = BlurImage(final_diff_inv_images[i], 5)

		if back:
			PlotImg(backsub_images[i], '%s - Background' % images[i].ImgName)
		if final_inv:
			PlotImg(final_inv_images[i], 'Inv Bin Thresh % s' % images[i].ImgName)
		if final_inv_blur:
			PlotImg(final_inv_images_blurred[i], 'Inv Bin Thresh Blur %s' % images[i].ImgName)
		if final_diff:
			PlotImg(final_diff_inv_images[i], 'Bin diff %s - %s' % (images[i].ImgName, "precedente"))
		if final_diff_blur:
			PlotImg(final_diff_inv_images_blurred[i], 'Bin diff %s - %s Blur' % (images[i].ImgName, "precedente"))
		if key:
			PlotImg(imwhitkeypoints_images[i], 'Keypoints %s' % images[i].ImgName)
		# cv2.imwrite('Keypoints%s.jpg'%images[i].ImgName, imwhitkeypoints_images[i])

	# faccio ROI7 - ROI1 ----> faccio il BLUR -----> faccio threshold (binaria e binaria inversa) -----> ne rifaccio il BLUR

	plt.show()