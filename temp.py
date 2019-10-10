import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral as sp
from laspy.file import File

def histogram(image):
    img =cv2.imread(image)
    gray_img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    blue_array=np.zeros((256,),dtype=int)
    green_array=np.zeros((256,),dtype=int)
    red_array=np.zeros((256,),dtype=int)
    gray_array=np.zeros((256,),dtype=int)
    
    print(type(img))
    print(img.shape)
    print(img.shape[0])
    print(gray_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            blue_array[img[i,j][0]]=(blue_array[img[i,j][0]])+1
            green_array[img[i,j][1]]=(green_array[img[i,j][1]])+1
            red_array[img[i,j][2]]=(red_array[img[i,j][2]])+1
            
    fig=plt.figure()
    plt.title("Gray level intensity")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    for z in range(256):
        gray_array[z]=(blue_array[z]+green_array[z]+red_array[z])/3
        plt.scatter(z,gray_array[z], s=10)
    plt.show()
    fig.savefig('gray_intensity_binary.jpg')
    
    fig1=plt.figure()
    plt.title("Blue intensity")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    for k in range(256):
        plt.scatter(k,blue_array[k], s=10)
    plt.show()
    fig1.savefig('blue_binary.jpg')
    
    fig2=plt.figure()
    plt.title("Green intensity")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    for k in range(256):
        plt.scatter(k,green_array[k], s=10)
    plt.show()
    fig2.savefig('green_binary.jpg')
    
    fig3=plt.figure()
    plt.title("Red intensity")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    for k in range(256):
        plt.scatter(k,red_array[k], s=10)
    plt.show()
    fig3.savefig('red_binary.jpg')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def binary_image(image1,image2):
    count=0
    img1=cv2.imread(image1)
    img2=cv2.imread(image2)
    image3=img1-img2
    (thresh_img,x)=cv2.threshold(image3,55,255,cv2.THRESH_BINARY)
    cv2.imwrite('binary_image_vegetation.jpg',x)
    print(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i,j][0]>250):
                count=count+1
                x[i,j][0]=1
                x[i,j][1]=1
                x[i,j][2]=1
            else:
                x[i,j][0]=0
                x[i,j][1]=0
                x[i,j][2]=0
    #histogram('binary_image.jpg')
    print(count)
    plt.imshow(image3)
    
def hyperspectral(image):
    img=sp.open_image(image)
    view=sp.imshow(img,(4,3,2))
    sp.save_rgb('false_color.jpg',img,(4,3,2))
    print(img.shape)
    print(view)
    print(img)
    red=img[:,:,2]
    nir=img[:,:,3]
    ndvi=((nir-red)/(nir+red+0.00001))
    sp.imshow(ndvi)
    sp.save_rgb('ndvi.jpg',ndvi)
    sp.imshow(img,(6,6,0))
    
def lidar(image):
    las=File(image,mode="r")
    for p in las:
        print(las.x,las.y,las.z)
    point=las.points
    print(point)
    print(las.header.min)
    print(las.header.max)
    print(las.x)
    print(las.X)
    print(las.Y)
    
    
histogram('sunset.jpg')
#binary_image('false_color.jpg','ndvi.jpg')
#hyperspectral('tipjul1.lan')
#lidar('17258975.las')