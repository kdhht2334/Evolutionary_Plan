__author__ = "kdhht5022@gmail.com"
"""
Do make sure: opencv-python: 3.4.1 

using following cmd:
    $pip install opencv-python==3.4.1.0
"""
import numpy as np
import time
from multiprocessing import Queue, Value
import configparser, uuid

import asyncio

import cv2
from fabulous.color import fg256

from my_util.videomgr import VideoMgr


async def async_handle_video(camInfo):
    videoSaveOut = None
    config = camInfo["conf"]    
    VideoHandler = VideoMgr(int(config['url']), config['name'])        
    VideoHandler.open(config)        
   
    loop = asyncio.get_event_loop()

    global key
    global image_queue
    global cam_check
    
    global do_hand_gesture
    
    def sleep():
        time.sleep(0.02)
        return 
    def resize(img, size):
        return cv2.resize(img, size)
    try:
        f = 0  # just for counting :)
        while(True):
            if cam_check[cname[VideoHandler.camName]] == 0:
                ret, orig_image = await loop.run_in_executor(None, VideoHandler.camCtx.read)
                if cname[VideoHandler.camName]%2 == 1:
                    orig_image = np.fliplr(orig_image)
                    
#                    cv2.imwrite('/home/kdh/Desktop/smartstore/vids/samples_{}.png'.format(f), orig_image)

                # Trigger using some condition
                if f % 10 == 0:
                    do_hand_gesture = True
                    
#                orig_image = await loop.run_in_executor(None, resize, orig_image[:,500:], (400,400))
#                orig_image_small = await loop.run_in_executor(None, resize, orig_image, (200,200))
                
#                if cname[VideoHandler.camName] < 4:
#                    image_queue.append([cname[VideoHandler.camName], orig_image, orig_image_small])
#                    cam_check[cname[VideoHandler.camName]] = 1
#                if len(image_queue) > 50:
#                    image_queue[:20] = []

                if (config['camshow'] == 'on'):# & (cname[VideoHandler.camName]%2 == 0 ) & (False):
                    cv2.imshow(config['name'], orig_image)
#                    cv2.imwrite('/home/kdh/Desktop/smartstore/vids/samples_{}.png'.format(f), orig_image)
                    if VideoHandler.camName == '1th_left':
                        key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                f += 1
                    
            else:
                await loop.run_in_executor(None, sleep)
                if sum(cam_check) == 4:
                    cam_check = [0,0,0,0]


        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()

    except asyncio.CancelledError:        
        if config['tracking_flag'] == 'off' and config['videosave'] == 'on':        
            videoSaveOut.release()
            
        cv2.destroyAllWindows()
        VideoHandler.camCtx.release()


async def handle_video_analysis():    
    def hand_detection():

        global hand_gesture_config
        global hand_gesture_sleep
        global do_hand_gesture
        
        # execute `hand gesture` when the signal is activated.
        if do_hand_gesture == 1:
            
            print(fg256("cyan", "[INFO] Do hand gesture!"))
            
#            img_height   = 400
#            img_width    = 400
#            
#            # 2) Prepare prediction
#            batch_images = np.array(top_img)
#            batch_images = batch_images[:,:,:,(2,1,0)]
#            
#
#
#            y_pred = detection_end_point.predict(batch_images)
#
#            y_pred_decoded_inv = decode_detections(y_pred,                 # original coefficients
#                                                   confidence_thresh=0.6,  # confidence_thresh=0.55
#                                                   iou_threshold=0.7,      # iou_threshold=0.4
#                                                   top_k=300,              # top_k=200
#                                                   normalize_coords=True,
#                                                   img_height=img_height,
#                                                   img_width=img_width)
#
#            product = step3_modules.final_pred(y_pred_decoded_inv)

            do_hand_gesture = False
        else:
            hand_gesture_sleep = True

    loop = asyncio.get_event_loop()
    try:
        while(True):
            await loop.run_in_executor(None, hand_detection)
            if key == ord('q'):
               break
    except asyncio.CancelledError:        
        pass
        
        
# agent algorithm
async def Agent():
    global action

    def agent():
        
        print(fg256("orange", "[INFO] Main Loop"))

        global open_algorithm
        global action
        global something
        global key
        
        if open_algorithm == 1:
            something = time.time()
        if something > 0:
            if time.time() - something > 0.8:
                something = 0
                action = 'something'

          
        # re-initialize global variables
        time.sleep(2e-2)
        action = None

    loop = asyncio.get_event_loop()
    try:
        while(True):
            await loop.run_in_executor(None, agent)
            if key == ord('q'):
               break

    except asyncio.CancelledError:        
        pass


async def async_handle_video_run(camInfos):  
    futures = [asyncio.ensure_future(async_handle_video(cam_info)) for cam_info in camInfos]\
             +[asyncio.ensure_future(handle_video_analysis())]\
             +[asyncio.ensure_future(Agent())]

    await asyncio.gather(*futures)


class Config():
    """  Configuration for Label Convert Tool """
    def __init__(self):
        global ini             
        self.inifile = ini
        self.ini = {}
        self.debug = False
        self.camera_count = 0
        self.cam = []        
        self.parser = configparser.ConfigParser()
        self.set_ini_config(self.inifile)        
               
    def set_ini_config(self, inifile):
        self.parser.read(inifile)
        
        for section in self.parser.sections():
            self.ini[section] = {}
            for option in self.parser.options(section):
                self.ini[section][option] = self.parser.get(section, option)
            if 'CAMERA' in section:
                self.cam.append(self.ini[section])


"""
 *******************************************************************************
 * [CLASS] FER Integration Application
 *******************************************************************************
"""
class FER_INT_ALG():
    """ CVIP Integrated algorithm """    
    def __init__(self):
        global ini
        ini = 'config.ini'
        self.Config = Config()
        self.trackingQueue = [Queue() for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count']))]        
        self.vaQueue = Queue()
        self.isReady = Value('i', 0)
        self.camTbl = {}

        global open_algorithm
        global something; global action
        
        global key
        global image_queue; global image_queue_move
        global cam_check
        
        global cname

        global do_hand_gesture
        global hand_gesture_config
        global hand_gesture_sleep
        
        open_algorithm = True
        something = 0; action = None
        do_hand_gesture = False

        hand_gesture_sleep = True

        image_queue = []
        image_queue_move = []
        key = [0,0,0,0]

        # We can set multiple cameras in the future!
        cname = {'1th_left'  : 0,}
#                 '1th_right' : 1,}

        cam_check = [0,0,0,0]
        
#        global end_point
#        global detection_end_point
#        detection_end_point = step3_modules.init_detection()
#        end_point = init_net(5)
        
    def run(self):        
        camInfoList = []        
        camTbl = {}        
        global key
#        img_data = [[]]*int(self.Config.ini['COMMON']['camera_count'])

        for idx in range(0, int(self.Config.ini['COMMON']['camera_count'])):
            camInfo = {}
            camUUID = uuid.uuid4()
            
            camInfo.update({"uuid": camUUID})
            camInfo.update({"isready": self.isReady})
            camInfo.update({"tqueue": self.trackingQueue[idx]})
            camInfo.update({"vaqueue": self.vaQueue})            
            camInfo.update({"conf": self.Config.cam[idx]})
            camInfo.update({"globconf": self.Config})
            
            camInfoList.append(camInfo)
            camTbl.update({camUUID: camInfo})
        
        while (True):
            loop = asyncio.get_event_loop()

            # step1 (mult-camera processing) & step2 (top_k selector) 
            # & step3 (top_k frame prediction using SSD)
            loop.run_until_complete(async_handle_video_run(camInfoList))
            loop.close()
            if key == ord('q'):
                break
                

    def close(self):
        for idx in range(0, int(self.Config.ini['COMMON']['camera_proc_count'])):
            self.trackingQueue[idx].close()
            self.trackingQueue[idx].join_thread()
        self.vaQueue.close()
        self.vaQueue.join_thread()
        #self.VideoHdlrProc.stop()
        #self.AnalyzerHdlrProc.stop()
        #self.TrackingHdlrProc.stop()
        

if __name__ == "__main__":
    fer_int_alg = FER_INT_ALG()
    print ('Start... FER application')
    fer_int_alg.run()
    fer_int_alg.close()
    print("Completed FER application")









