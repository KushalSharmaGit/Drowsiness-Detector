prediction = model.predict(eye)
       # print(prediction)
       #Condition for Close
        if prediction[0][0]>0.30:
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            score=score+1
            #print("Close Eyes")
            if(score > 20):
                try:
                    sound.play()
                except:  # isplaying = False
                    pass

        #Condition for Open
        elif prediction[0][1] > 0.70:
            score = score - 1
            if (score < 0):
                score = 0
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            #print("Open Eyes")
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
