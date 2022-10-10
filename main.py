import cv2
from FaceDetector import CenterFace
from Classifier import Inference

def main():
    infer = Inference()
    detector = CenterFace() 
    
    video = cv2.VideoCapture('input.mp4')
    if (video.isOpened() == False): 
        print("Error reading video file")
        exit()
        
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
   
    frame_size = (frame_width, frame_height)
    
    saved_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'),10, frame_size)
    label_color = {0: (0,0,255), 1: (0,255,0)}
    
    while(True):
        ret, original_img = video.read()
        
        if not ret:
            break
        
        nface, bounding_boxs, lmks = detector.detect_faces(original_img)
        if nface > 0:
            for box in bounding_boxs:
                pred = infer.run_inference(original_img, box)
                cv2.rectangle(original_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), label_color[pred], 2)
                
        saved_video.write(original_img)
        cv2.imshow('Show', original_img)
        key = cv2.waitKey(1)
    
        if key == 27:
            break
        
    video.release()
    saved_video.release()
    cv2.destroyAllWindows()
   
   
if __name__ == "__main__": 
    main()

    
    