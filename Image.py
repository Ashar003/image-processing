#Akash Sharma
from​ ​__future__​ ​import​ print_function
from​ PIL ​import​ Image ​#module for importing from​ matplotlib ​import​ pyplot ​as​ plt
from​ scipy.ndimage ​import​ filters
import​ numpy ​as​ np
import​ argparse
import​ cv2
#try/error
try​:
img = cv2.imread(​'/Users/amrahs/Desktop/h/img1.jpg'​ ) img2 = cv2.imread(​'/Users/amrahs/Desktop/h/img2.jpg'​ )
except​ ​IOError​: ​pass
​#convert to gray
gray1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) gray2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) cv2.imwrite(​'gray1.jpg'​,gray1) cv2.imwrite(​'gray2.jpg'​,gray2)
#to show image
# cv2.imshow(img)
# cv2.imshow(img2)
# cv2.imshow('Gray
# cv2.imshow('Gray #cv2.waitKey(0) #cv2.destroyAllWindows() #print(img,img2)
#1. Capture two images, that will be used for processing, (one underexposed, and one overexposed) using your cell phone or digital camera and generate their corresponding gray level images (e.g., gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).
#• Apply the gamma transformation to these two gray level images to correct their #appearance.
#• Test several parameters of gamma until obtaining the best results and plot the
image', gray1)
image', gray2)
 #histograms of both original and corrected images.
#adjust gamma
def​ ​adjust_gamma​(​gray1​, ​gamma​=​1.0​): ​#function to adjust gamma
​# build a lookup table mapping the pixel values [0, 255] to ​# their adjusted gamma values
invGamma = ​1.0​ / gamma
table = np.array([((i / ​255.0​) ** invGamma) * ​255
​for​ i ​in​ np.arange(​0​, ​256​)]).astype(​"uint8"​) ​# lookup table to make gamma correction
​return​ cv2.LUT(gray1, table)
for​ gamma ​in​ np.arange(​0.0​, ​3.5​, ​0.5​):
​# ignore when gamma is 1 (there will be no change to the image) ​if​ gamma == ​1​:
​continue gamma = ​0.1
#5.0 good for img2
#higher than 1 is washed out img1 0.3
​# apply gamma correction and show the images gamma = gamma ​if​ gamma > ​0​ ​else​ ​0.1
adjusted = adjust_gamma(gray1, ​gamma​=gamma)
cv2.imwrite(​'gray1A.jpg'​,gray1)
for​ gamma ​in​ np.arange(​0.0​, ​3.5​, ​0.5​):
​# ignore when gamma is 1 (there will be no change to the image)
# # #
​if​ gamma == ​1​: ​continue
gamma= ​5.0
​#5.0 good for img2
​# apply gamma correction and show the images gamma = gamma ​if​ gamma > ​0​ ​else​ ​0.1 adjusted2 = adjust_gamma(gray2, ​gamma​=gamma)
cv2.imshow("Images", np.hstack([gray1, adjusted2])) cv2.imshow("Images", np.hstack([gray2, adjusted2]))
cv2.destroyAllWindows()

 cv2.imwrite(​'gray2A.jpg'​,gray2)
hist1 = cv2.calcHist([gray1],[​0​],​None​,[​256​],[​0​,​256​]) hist1o = cv2.calcHist([img],[​0​],​None​,[​256​],[​0​,​256​]) hist2 = cv2.calcHist([gray2],[​0​],​None​,[​256​],[​0​,​256​]) hist2o = cv2.calcHist([img2],[​0​],​None​,[​256​],[​0​,​256​])
plt.plot(hist1) plt.plot(hist1o) plt.plot(hist2) plt.plot(hist2o) plt.show()
# 2. Apply Histogram equalization to the two images captured previously. You can use build- in functions like
#•
#equalize
Eq1 = cv2.equalizeHist(img) Eq2 = cv2.equalizeHist(img2)
#plot
plt.plot(Eq1) plt.plot(Eq2)
#show image
# cv2.imshow(Eq1)
# cv2.imshow(Eq2) cv2.imwrite(​'gray1E.jpg'​,Eq1) cv2.imwrite(​'gray2E.jpg'​,Eq2)
# 3. Implement the algorithm of exact histogram matching using the following kernels: #01011 1
# cv2.equalizeHist(img).
# Show resulting images and their histograms.
# 11 w=[1] w2= 111 w3= 111

 # 159 0 1 0 1 1 1
#
# • You can use build-in functions like cv2.filter2D(gray, -1, kernel) for implementing the convolutions.
# • Use as reference for the output histogram a uniform distributed function, and apply your algorithm to the two images used previously for processing.
# • Show the resulting images and their corresponding histograms.
# • Compare the results obtained in (1), (2), and (3).
#Kernels
kernel0 = arr.array([​1​])
kernelT = (​1​/​5​) * arr.array([​0​,​1​,​0​ ],[​1​,​1​,​1​],
[​0​,​1​,​0​])
kernelH = (​1​/​9​) * arr.array([​1​,​1​,​1​],
[​1​,​1​,​1​] [​1​,​1​,​1​])
#apply each of the kernel to images
imgO1= cv2.filter2D(gray1, -​1​, kernelO) imgT1= cv2.filter2D(gray1, -​1​, kernelT) imgH1=cv2.filter2D(gray1, -​1​, kernelH)
imgO2= cv2.filter2D(gray2, -​1​, kernelO) imgT2= cv2.filter2D(gray2, -​1​, kernelT) imgH2=cv2.filter2D(gray2, -​1​, kernelH)
#show
# cv2.imshow(imgO1) # cv2.imshow(imgT1) # cv2.imshow(imgH1)
cv2.imwrite(​'grayO1.jpg'​,imgO1) cv2.imwrite(​'grayT1.jpg'​,imgT1) cv2.imwrite(​'grayH1.jpg'​,imgH1)
# cv2.imshow(imgO2) # cv2.imshow(imgT2) # cv2.imshow(imgH2)

 cv2.imwrite(​'grayO2.jpg'​,imgO2) cv2.imwrite(​'grayT2.jpg'​,imgT2) cv2.imwrite(​'grayH2.jpg'​,imgH2)
#histogram
histO1 = cv2.calcHist([imgO1],[​0​],​None​,[​256​],[​0​,​256​]) histT1 = cv2.calcHist([imgT1],[​0​],​None​,[​256​],[​0​,​256​]) histH1 = cv2.calcHist([imgH1],[​0​],​None​,[​256​],[​0​,​256​])
histO2 = cv2.calcHist([imgO2],[​0​],​None​,[​256​],[​0​,​256​]) histT2 = cv2.calcHist([imgT2],[​0​],​None​,[​256​],[​0​,​256​]) histH2 = cv2.calcHist([imgH2],[​0​],​None​,[​256​],[​0​,​256​])
#plot
plt.plot(imgO1) plt.plot(imgT1) plt.plot(imgH1)
plt.plot(imgO2) plt.plot(imgT2) plt.plot(imgH2)
# 4. Select one image that was previously improved, and apply to this image the following
# operators:
# a. Smoothing spatial filtering (Gaussian and Box Kernels)
# b. First-order derivative (Robert and Sobel Kernels)
# c. Second-order derivative
# d. Unsharp and Highboost filtering
# • Show the images obtained after using the previous operators.
#Gaussian Blur and deviation and img
blur = cv2.GaussianBlur(img,(​5​,​5​),​0​) # cv2.imshow(blur) #show

cv2.imwrite(​'blur.jpg'​,blur)
#Second-order deriv
laplacian = cv2.Laplacian(img,cv2.CV_64F) ​#laplacian with option to alter image # cv2.imshow(laplacian)
cv2.imwrite(​'laplacian.jpg'​,laplacian)
#Sobel
imx = zeros(img.shape)
filters.sobel(img,​1​,imx)
subplot(​1​,​4​,​2​)
axis(​'off'​)
cv2.imwrite(​'imx.jpg'​,imx)
#Unsharp using gray image minus a certain multiplier times laplacian
sharp = gray1- ​0.7​*laplacian
# cv2.imshow(sharp) #show cv2.imwrite(​'sharp.jpg'​,sharp)