# !pip install ultralytics==8.0.88 # 실시간은 8.0.88 버전에서 가능 

import cv2
from ultralytics import YOLO
import os
from multiprocessing import Process
from utils import video_1_load_video, video_2_yolo_1, gps_1_find_nearest_school, led_1, alram_2_result_analysis, led_2_beep


'''
NOTE : module pipeline

utils.py의 모듈들을 순서에 맞게 배치
(각 모듈들의 설명은 utils 페이지에서 확인)

'''
source_path = os.getcwd()
window_name = 'combined'

def implement(source_path, source_1, source_2, latitude, longitude):
    
    print('✅ source_path : ', source_path)
    
    cap1, video_info_1 = video_1_load_video(source_path, source_1)
    cap2, video_info_2 = video_1_load_video(source_path, source_2)
    ori_w_1, ori_h_1 = video_info_1.resolution_wh
    ori_w_2, ori_h_2 = video_info_2.resolution_wh

    model1 = YOLO(('yolov8l.pt'))
    model2 = YOLO(('yolov8l.pt'))

    frame_size=(500, 500)    
    frame_h = frame_size[0] # 세로
    frame_w = frame_size[1] # 가로

    frame_size_led = (500, 700)
    frame_h_led = frame_size_led[0] 
    frame_w_led = frame_size_led[1]

    video_out_path = os.path.join(source_path, 'output', f'{source_1}_{source_2}.mp4') 
    cap_out = cv2.VideoWriter(video_out_path, 
                            cv2.VideoWriter_fourcc(*'MP4V'), 
                            cap1.get(cv2.CAP_PROP_FPS),
                            (frame_w * 2, frame_h))    
    
    car_path = os.path.join(source_path, 'led_source', 'car.png')
    person_path = os.path.join(source_path, 'led_source', 'person.png')
    warn_path = os.path.join(source_path, 'led_source', 'warn.png')
    crash_path = os.path.join(source_path, 'led_source', 'crash.png')

    car_img = cv2.imread(car_path)
    person_img = cv2.imread(person_path)
    crash_img = cv2.imread(crash_path)
    warn_img = cv2.imread(warn_path)

    imogi_ratio_w = int(frame_w_led * 0.3)
    imogi_ratio_h = int(frame_h_led * 0.5)

    car_img = cv2.resize(car_img, (imogi_ratio_w, imogi_ratio_h)) # w : 320, h : 480    
    person_img = cv2.resize(person_img, (imogi_ratio_w, imogi_ratio_h)) 

    crash_img = cv2.resize(crash_img, (int(frame_w_led * 0.2), int(frame_h_led * 0.4))) 
    warn_img = cv2.resize(warn_img, (int(frame_w_led * 1), int(frame_h_led * 0.4))) # (800, 320)

    imgs = [car_img, person_img, crash_img, warn_img]

    warning_count_down = 0
    alraming_time = 100
    c = 0
    warning_count = 5
    warning_text1 = ''
    warning_text2 = ''
    real_warning_text1 = ''
    real_warning_text2 = ''
    before_track_infos1 = []
    before_track_infos2 = []
    detect_type='weight'
    weight = 4

    school_info = gps_1_find_nearest_school(latitude, longitude, source_path)

    while True:
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        result1, result2 = video_2_yolo_1(model1, model2, frame1, frame2)

        video_1 = [result1, ori_w_1, ori_h_1, video_info_1, model1, frame1, before_track_infos1, real_warning_text1]
        video_2 = [result2, ori_w_2, ori_h_2, video_info_2, model2, frame2, before_track_infos2, real_warning_text2]
        utils = [detect_type, warning_count, weight, alraming_time, c, warning_count_down]
        
        
        cap_out, real_warning_text_set, before_track_infos_set, text_source_set, zone_set = alram_2_result_analysis(video_1, video_2, cap_out, utils, frame_size)
                
        real_warning_text1 = real_warning_text_set[0]
        real_warning_text2 = real_warning_text_set[1]

        before_track_infos1 = before_track_infos_set[0]
        before_track_infos2 = before_track_infos_set[1]

        utils_led = [alraming_time, c, warning_count_down]
        text_set = [warning_text1, warning_text2]
        
        real_warning_text_set, warning_count_down = led_1(source_path, real_warning_text1, real_warning_text2, imgs, school_info, frame_size_led, text_source_set, zone_set, utils_led, text_set)
        
        real_warning_text1 = real_warning_text_set[0]
        real_warning_text2 = real_warning_text_set[1]

        if (warning_count_down > 0) & (0 == (warning_count_down % 5)):
            p_led = Process(target=led_2_beep(source_path))
            p_led.start()
        
        # 키 입력을 1밀리초 대기하고, 'q' 키가 눌리면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # 반복문이 종료되면, 모든 창을 닫고 비디오 캡처를 해제
    cv2.destroyAllWindows()
    cap1.release()
    cap2.release()
    cap_out.release()



##############################################################################

source_1 = 'blur_person.mp4'
source_2 = 'blur_car.mp4'



# 초등학교 위경도 좌표 입력
latitude = 35.00000
longitude = 129.00000

implement(source_path, source_1, source_2, latitude, longitude)