import numpy as np
import cv2
import imutils
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# İmage dosyasını okuduk
image = cv2.imread('Car Images/img1.jpeg')

# Resmi yeniden boyutlandırdık - genişliği 500 olarak değiştiriyor
image = imutils.resize(image, width=500)

# Orijinal görüntüyü gösterildi
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# RGB'den Gri tonlamaya dönüştürme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.waitKey(0)

# Yinelemeli ikili filtre ile gürültü giderme (kenarları korurken gürültüyü giderir)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.waitKey(0)

# grayscale görüntünün Kenarlarını Bulduk
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edges", edged)
cv2.waitKey(0)

# Kenarlara göre konturları bulduk
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Tüm konturları çizmek için orijinal görüntünün bir kopyası oluşturuldu
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4- All Contours", img1)
cv2.waitKey(0)

#minimum gerekli alanı '30' olarak tutarak konturları alanlarına göre sıralayın (bundan daha küçük herhangi bir şey dikkate alınmayacaktır)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None #we currently have no Number plate contour

# Top 30 Contours
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 30 Contours", img2)
cv2.waitKey(0)

# plakanın mümkün olan en iyi yaklaşık konturunu bulmak için konturlarımız üzerinde dolaştık
count = 0
idx =7
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour

            # Bu konturları kırpın ve Kırpılmış Görüntüler klasöründe saklayın
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w] #Create new image
            cv2.imwrite('Cropped Images-Text/' + str(idx) + '.png', new_img) #Store new image
            idx+=1

            break


# Seçilen konturun orijinal görüntü üzerine çizilmesi
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.waitKey(0)

Cropped_img_loc = 'Cropped Images-Text/7.png'
cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

# Görüntüyü dizgeye dönüştürmek için tesseract kullanın
text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("Number is :", text)

cv2.waitKey(0) #Gösterilen resimleri kapatmadan önce kullanıcı girişi için bekleyin
