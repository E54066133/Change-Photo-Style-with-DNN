from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import RPi.GPIO as GPIO


#設定腳位
button = 3  #按鈕腳位

#定義模式
GPIO.setmode(GPIO.BOARD)
GPIO.setup(button, GPIO.IN)

#定義相機物件
camera = PiCamera()
camera.resolution = (320, 240)    #定義解析度(最大可支援到3280×2464)


try:
    while True:
        if GPIO.input(button) == GPIO.HIGH :
            print("pressed!!")
            localtime = time.localtime()                                    #紀錄目前時間
            temp = time.strftime("%Y/%m/%d/%H/%M/%S %p", localtime)         #暫存時間位元組
            data = temp.split("/")                                          #字串切割
            name = data[1]+data[2]+'_'+data[3]+data[4]+'.jpg'               #用時間來命名照片
            with PiRGBArray(camera) as output:                              #呼叫相機物件，並擷取單張照片
                #type1: save image directly
                #camera.capture(name)

                #type2: save image as opencv format (nparray)
                camera.capture(output, format='bgr')
                img = output.array                                          #影像(img)在此被儲存為array的格式，可以接續讓opencv做處理
                cv2.imwrite(name, img)                                      #這裡直接將img以opencv的function儲存
                
                #處理風格轉換
                #讀取風格檔，這裡讀入"星空"的風格
                name_tran =  data[1]+data[2]+'_'+data[3]+data[4]+'_tran'+'.jpg'
                net = cv2.dnn.readNetFromTorch('model/starry_night.t7')     #匯入"星空"風格轉換檔
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)        #用 DNN 運算

                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(img, 1.0, (w,h), (103.939, 116.779, 123.680), swapRB=False, crop=False)    #把影像修改成神經網路可以使用的格式

                
                start = time.time()                                         #紀錄開始時間
                net.setInput(blob)                                          #把影像丟入模型做風格轉換
                out = net.forward()                                         #開始轉換!
                out = out.reshape(3, out.shape[2], out.shape[3])            #調整out矩陣陣列形式
                out[0] += 103.939                                           #調色(參考網路)
                out[1] += 116.779                                           #調色(參考網路)
                out[2] += 123.68                                            #調色(參考網路)
                out = out.transpose(1, 2, 0)                                #轉置圖層
                end = time.time()                                           #紀錄結束時間
                
                
                print("computation time = {} sec.".format(end-start))

                #cv2.imshow("a", out)
                cv2.imwrite(name_tran, out)   #這裡直接將img以opencv的function儲存
                cv2.waitKey(0)
                #備註 : 如果58行和60行衝突 -> 把處理風格轉換 36行~58行 移出 try 去做處理
                
except KeyboardInterrupt:
    print('interrupt')
finally:
    camera.close()                 #關閉相機
    cv2.destroyAllWindows()        #關閉所有視窗
