# !pip install ultralytics==8.0.88 # 실시간은 8.0.88 버전에서 가능 

import cv2
import supervision as sv
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
from geopy.distance import geodesic
import pandas as pd
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play


'''
NOTE : 모듈 설명

1. Video
- YOLO를 통해 분석된 frame의 detection, tracking 정보 수집
- 수집된 정보를 통해 동적, 정적 객체의 구분

2. Alram
- 알람 조건 정의
- Video 모듈로 부터 수집, 획득한 정보를 통해 알람 조건과 대조 후 알람 신호 발생

3. LED
- Alram 모듈에서 전달받은 신호를 그래픽으로 띄움

4. GPS
- 위도, 경도좌표를 학교 정보(csv)에서 검색 후 LED모듈에 학교명 전달

'''


# Video 1 : 소스영상 불러오기
# sorce video를 지정한 경로에서 불러옴
def video_1_load_video(source_path, source_name):

    if source_name == 0:
        source_path = 0
    
    elif source_name == 1:
        source_path = 1

    else:
        source_path = os.path.join(source_path, "source", source_name)
        
    return cv2.VideoCapture(source_path), sv.VideoInfo.from_video_path(source_path)




# Video 2 : 객체 탐지, 추적 결과 directory 반환
# YOLO모델이 Detction, tracking한 결과를 반환
def video_2_yolo_1(model1, model2, frame1, frame2):
    results_m1 = model1.track(source=frame1, show=False, stream=True, device='mps')
    results_m2 = model2.track(source=frame2, show=False, stream=True, device='mps')

    return results_m1, results_m2





# Video 3 : 반환된 탐지, 추적 결과 directory에서 return_label, frame, detections  추출
# class 아이디, track 아이디, bounding box 좌표
def video_3_yolo_2_zone(model, result, frame):

    detections = sv.Detections.from_yolov8(result)
    
    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.numpy().astype(int)

    bbox = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id
    track_id = detections.tracker_id

    try:
        return_label = [
            [bbox[idx], 0 , confidence[idx], class_id[idx], track_id[idx], model.model.names[class_id[idx]]] for idx in range(len(result))] 
    except:
        pass
    
    return return_label, frame, detections





# Video 4 : 객체별 알람을 위한 첫번째 조건 반환
# 조건 정의 : 객체별 bbox의 좌측 상단 좌표가 움직인 거리들의 평균
def video_4_first_track(return_label, before_track_infos, target_classes, sequence): 
    
    for class_name in target_classes:  
        globals()[f'cond_list_{class_name}'] = [] 
    
    for info in return_label:
        x1, y1, x2, y2 = info[0]
        track_id = info[4]
        class_name = info[5]

        # 좌측상단(Left, Up) 좌표
        if sequence == 'left':        
            bbox_point = (x1,y2)
        elif sequence == 'right':        
            bbox_point = (x2,y2)

        # Compare to Before : 추적된 객체가 있다는 가정 하에 진행
        # before_track_infos : bbox_point, track_id, class_id
        if len(before_track_infos) > 0:

            for info in before_track_infos:

                #if track_id == info[1]:
                if (track_id == info[2]) & (class_name in target_classes):
                        bx, by = info[0]
                        x, y = bbox_point

                        # Computing Uclidian Distance Between Before and Now
                        ud = np.sqrt((bx - x)**2 + (by - y)**2)

                        cond = ud 
                        globals()[f'cond_list_{class_name}'].append(cond)
    
    # define mean of Uclidian Distances
    cond_mean_dict = dict()
    try:
        for class_name in target_classes:
            cond_mean_dict[class_name] = np.mean(globals()[f'cond_list_{class_name}']) 
    except:
        pass
    
    return cond_mean_dict





# Alram 1 : 알람 조건에 따라 알람 발생
def alram_1_second_track(frame, sequence, return_label, cond_mean_dict, detect_type, before_track_infos, warning_count, weight=1):
    
    warning_text = ''

    # make list for recording frame's tracking infos
    tracks_info = []
    track_ids = np.array([i[4] for i in return_label])
    class_names = np.array([i[5] for i in return_label])

    # for track 2️
        
    for info in return_label:
        x1, y1, x2, y2 = info[0]
        box_area = info[1]
        track_id = info[4]
        class_name = info[5]         

        try:
            cond_list_mean = cond_mean_dict[class_name]
        except:
            cond_list_mean = 10000

        # center 
        if sequence == 'left':        
            bbox_point = (x1,y2)
        elif sequence == 'right':        
            bbox_point = (x2,y2)
                
        # record tracks_info
        info = [bbox_point, box_area, track_id]
        tracks_info.append(info)

        for info in before_track_infos:
            if track_id == info[2]:                        

                # Computing Uclidian Distance
                bx, by = info[0]
                x, y = bbox_point
                ud = np.sqrt((bx - x) ** 2 + (by - y) ** 2)
                cond = ud
                
                # 차에 비해서 사람은 Identifying Moving Object이 잘 안됨. 따라서 가중치 부여
                if ('person' in class_names) & ('car' in class_names):
                    if class_name == 'person':
                        cond = (cond + weight) ** 2
                    if class_name == 'car':
                        cond = cond // 3

                if detect_type == 'weight':
                    cond_ud = (cond > cond_list_mean * weight)
                elif detect_type == 'square':
                    cond_ud = (cond ** weight > cond_list_mean ** weight)
                elif detect_type == 'exp':
                    cond_ud = (np.exp(cond) > np.exp(cond_list_mean) * weight)
                elif detect_type == 'exp2':
                    cond_ud = (np.exp2(cond) > np.exp2(cond_list_mean) * weight) 
                    
                elif detect_type == 'expm1':
                    cond_ud = (np.expm1(cond) > np.expm1(cond_list_mean) * weight)
                
                # Identifying Moving Object
                if cond_ud: 

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"MOVING_{track_id}_{class_name}",                                         
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                (0, 0, 255), 
                                thickness=2)
                    
                    print(f'A moving {class_name} detected!!!')
                    
                    # Moving Object에 WARNING 부여                            
                    try:
                        globals()[f'track_id_{track_id}_count_{sequence}'] += 1 
                    except:
                        globals()[f'track_id_{track_id}_count_{sequence}'] = 1 
                                                                                            
                    
                    # WARNING을 주는 기준 : warning_count
                    # warning 조건 달성
                    if globals()[f'track_id_{track_id}_count_{sequence}'] >= warning_count:
                        cv2.putText(frame, "WARNING", 
                                    (int(x1), int(y1) - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    2, (0, 165, 255), thickness=4) 
                        
                        print(f'Watch out for {class_name}!!!')

                        # 만약 하나의 track_id 에서 WARNING이 발생한다면 나머지는 0으로 초기화
                        for i in track_ids: 
                            if track_id == i: 
                                pass 
                            else: 
                                globals()[f'track_id_{track_id}_count_{sequence}'] = globals()[f'track_id_{track_id}_count_{sequence}'] * 1  // 3


                        # LED에 신호를 주는 변수 
                        if (class_name == 'car') | (class_name == 'motorbike') | (class_name == 'truck'): 
                            warning_text = 'car' 
                        elif (class_name == 'person') | (class_name == 'bicycle'): 
                            warning_text = 'person' 
                                                    
                    # warning 조건 미달성
                    else: 
                        pass
                        
                else: 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    cv2.putText(frame, f"Stopping{track_id}_{class_name}",                                         
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                (255, 0, 0), 
                                thickness=2)
            
    return [tracks_info, warning_text, frame]






# Alram 2 : 알람 조건 충족시 LED모듈에 전달할 정보 반환
def alram_2_result_analysis(video_1, video_2, cap_out, utils, frame_size):
    
    target_classes = ['car', 'person', 'motorbike', 'bicycle', 'truck']    

    result1 = video_1[0]
    ori_w_1 = video_1[1]
    ori_h_1 = video_1[2]
    video_info_1 = video_1[3]
    model1 = video_1[4]
    frame1 = video_1[5]
    before_track_infos1 = video_1[6]
    real_warning_text1 = video_1[7]
    
    result2 = video_2[0]
    ori_w_2 = video_2[1]
    ori_h_2 = video_2[2]
    video_info_2 = video_2[3]
    model2 = video_2[4]
    frame2 = video_2[5]
    before_track_infos2 = video_2[6]
    real_warning_text2 = video_2[7]

    detect_type = utils[0]
    warning_count = utils[1]
    weight = utils[2]


    for result1, result2 in zip(result1, result2):
        
        # Moving, Stopping Algorithm
        print('### Source 1 ###')
        
        polygon = np.array([
            [int(ori_w_1 * 3 / 10), int(ori_h_1 * 6 / 10)],
            [int(ori_w_1 * 1), int(ori_h_1 * 5 / 6)],
            [int(ori_w_1 * 1), int(ori_h_1 * 1)],
            [int(0), int(ori_h_1 * 1)],
            [int(0), int(ori_h_1 * 6.5 / 10)],
        ])

        sequence = 'left'
        return_label1, frame_1, detections_1 = video_3_yolo_2_zone(model1, result1, frame1)                 
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info_1.resolution_wh)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)
        zone.trigger(detections=detections_1)
        count_in_zone_1 = zone.current_count
        frame_1 = zone_annotator.annotate(scene=frame_1)       
        cond_mean_dict = video_4_first_track(return_label1, before_track_infos1, target_classes, sequence)
        before_track_infos1, warning_text1_source, frame_1 = alram_1_second_track(frame_1, sequence, return_label1, cond_mean_dict, 
                                                                detect_type, before_track_infos1, warning_count, weight)

        print('\n\n### Source 2 ###')
        
        polygon = np.array([
            [int(0), int(ori_h_2 * 6 / 10)],
            [int(ori_w_2 * 6.5 / 10), int(ori_h_2 * 2.5 / 10)],
            [int(ori_w_2 * 1 / 1), int(ori_h_2 * 2.5 / 10)],
            [int(ori_w_2 * 6.75 / 10), int(ori_h_2 * 1)],
            [int(0), int(ori_h_2 * 1)]
        ])
        
        sequence = 'right'
        return_label2, frame_2, detections_2 = video_3_yolo_2_zone(model2, result2, frame2)
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info_2.resolution_wh)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)
        zone.trigger(detections=detections_2)
        count_in_zone_2 = zone.current_count
        frame_2 = zone_annotator.annotate(scene=frame_2)
        cond_mean_dict = video_4_first_track(return_label2, before_track_infos2, target_classes, sequence)
        before_track_infos2, warning_text2_source, frame_2 = alram_1_second_track(frame_2, sequence, return_label2, cond_mean_dict,
                                                                detect_type, before_track_infos2, warning_count, weight)
            
        combined_frame = cv2.hconcat([cv2.resize(frame_1, frame_size), cv2.resize(frame_2, frame_size)])
        window_name = 'Combined'
        cv2.imshow(window_name, combined_frame)
        # 원하는 창 위치로 이동시킴
        x_pos = 700  # 원하는 x 좌표
        y_pos = 0  # 원하는 y 좌표
        cv2.moveWindow(window_name, x_pos, y_pos)
        cap_out.write(combined_frame)

        return cap_out, [real_warning_text1, real_warning_text2], [before_track_infos1, before_track_infos2], [warning_text1_source, warning_text2_source], [count_in_zone_1, count_in_zone_2]





# GPS 1 : 가까운 학교 찾기
def gps_1_find_nearest_school(latitude, longitude, source_path):

    file_path_school = os.path.join(source_path, '전국 초등학교.csv')
    df_school = pd.read_csv(file_path_school, encoding = 'cp949')
    nearest_distance = float('inf')
    nearest_school_info = None
    current_time = datetime.now()
    hour = current_time.hour

    #hour = 14
    if 8 <= hour < 9:
        time_info = '등교시간입니다'

    elif 14 <= hour < 15:
        time_info = '하교시간입니다'

    else:
        time_info = '근처, 서행하세요'

    for index, row in df_school.iterrows():
        school_name = row['학교명']
        lat = row['위도']
        lon = row['경도']

        distance = geodesic((latitude, longitude), (lat, lon)).meters

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_school_info = school_name

    return nearest_school_info, time_info





# LED 1 : 위험한 객체의 종류를 LED에 띄워줌
def led_1(source_path, real_warning_text1, real_warning_text2, imgs, school_info, frame_size, text_source_set, zone_set, utils_led, text_set):

    school_name = school_info[0]
    school_time = school_info[1]

    frame_h = frame_size[0] # 세로
    frame_w = frame_size[1] # 가로
    
    LED_IMG = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)

    fontpath = os.path.join(source_path, 'led_source' ,'NanumGothicBold.otf')
    img_pillow = Image.fromarray(LED_IMG) # img_pillow로 변환
    font = ImageFont.truetype(fontpath, 25)
    draw = ImageDraw.Draw(img_pillow, 'RGB')
    draw.text((int(frame_w * 0.275), int(frame_h * 0.05)), school_name, font=font, fill= (144, 238, 144))
    draw.text((int(frame_w * 0.5), int(frame_h * 0.05)), school_time, font=font, fill= (144, 238, 144))
    
    LED_IMG = np.array(img_pillow)

    car_img = imgs[0]
    person_img = imgs[1]
    crash_img = imgs[2]
    warn_img = imgs[3]

    imogi_ratio_w = int(frame_w * 0.3)
    imogi_ratio_h = int(frame_h * 0.5)
    
    margin = 0.05
    left_h_start = int(frame_h * 0.1)
    left_h_end = left_h_start + imogi_ratio_h
    left_w_start = int(frame_w * margin)
    left_w_end = left_w_start + imogi_ratio_w

    right_h_start = int(frame_h * 0.10)
    right_h_end = left_h_start + imogi_ratio_h
    right_w_end = int(frame_w * (1 - margin))
    right_w_start = right_w_end - imogi_ratio_w
    
    alraming_time = utils_led[0]
    c = utils_led[1]
    warning_count_down = utils_led[2]


    warning_text1_source = text_source_set[0]
    warning_text2_source = text_source_set[1]
    count_in_zone_1 = zone_set[0]
    count_in_zone_2 = zone_set[1]

    warning_text1 = text_set[0]
    warning_text2 = text_set[1]
    # 양쪽 모두 warning 발생 시 second_track1[2] 업데이트 됨
    if (warning_text1_source != '') & (warning_text2_source != '') & ((count_in_zone_1 > 0) & (count_in_zone_2 > 0)): # 두 영상에 모두 WARNING 신호를 줄 때
        c += 1
        warning_text1 = warning_text1_source
        warning_text2 = warning_text2_source
        print(f'c : {c}')
    
        try: 
            if cond1: # 알람이 울리는 중 : warning_count_down 업데이트 X
                pass
            else: # 알람이 마치면 : warning_count_down 업데이트 O
                warning_count_down = alraming_time
        except:
            warning_count_down = alraming_time
    
    if c == 1:
        real_warning_text1 = warning_text1
        real_warning_text2 = warning_text2

    # cond : 
    cond1 = ((alraming_time / 4) < warning_count_down) & (warning_count_down <= alraming_time)


    if cond1 :
        real_warning_text1 = real_warning_text1
        real_warning_text2 = real_warning_text2
        warning_count_down -= 1
    else:
        real_warning_text1 = warning_text1
        real_warning_text2 = warning_text2
    
    # 알림 주기(속도)
    n = 5
    cond2 = warning_count_down % (alraming_time // n) >= ((alraming_time // n) // 2)

    if (cond1) & (cond2) :
            #person_img.shape : (200, 180, 3)

        if real_warning_text1 == 'car':
            img = car_img
        elif real_warning_text1 == 'person':
            img = person_img
            
        LED_IMG[left_h_start : left_h_end, left_w_start : left_w_end] = img
            
        # right
        if real_warning_text2 == 'car': 
            img = car_img               
        elif real_warning_text2 == 'person':
            img = person_img

        LED_IMG[right_h_start : right_h_end, right_w_start : right_w_end] = img
        LED_IMG[int(frame_h * (0.1 + 0.1)):int(frame_h * (0.5 + 0.1)), int(frame_w * 0.4):int(frame_w * 0.6)] = crash_img
        LED_IMG[np.int_(frame_h * 0.6):, :] = warn_img
    

    else:        
        pass
    
    cv2.imshow('LED', LED_IMG)

    return [real_warning_text1, real_warning_text2], warning_count_down





# LED 2 : beep 사운드
def led_2_beep(source_path):

    source_path = os.path.join(source_path, "audio/beep.mp3")
    sound = AudioSegment.from_mp3(source_path)
    play(sound)