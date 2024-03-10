# 🚸 스쿨존 충돌사고 예방 알리미.V2  
(School_Zone_Collision_Accident_Prevention_Notifier)
      
|구분|설명|
|------|---|
|프로젝트 명|딥러닝 기반 스쿨존(골목길) 충돌사고 예방 알고리즘 개발|
|제안배경|매년 높아지는 스쿨존 교통사고를 예방하기 위한 아이디어입니다.|
|요약|Object Detection 알고리즘인 YOLOv8을 base로 사용했습니다.<br>객체를 탐지 후 움직임을 탐지하는 알고리즘과 알림을 주는데 필요한 조건들을 추가해서<br>충돌사고를 예방할 수 있는 알리미를 개발했습니다.|

스택 적기 !!! 파이토치 욜로 등등

**시연영상**  👉 [https://youtu.be/BCXjsv-tun4](https://www.youtube.com/watch?v=Gpa8G2YvxPw)

<br/>
<br/>

### **1. 움직이는 객체와 정지해있는 객체를 분리합니다.**   
![image](https://github.com/CodeofO/School_Zone_V2/assets/99871109/bce880f8-0e7a-45ca-9d4f-32eaccdfea74)

![image](https://github.com/CodeofO/2023_06_V2_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/66b20d8d-6d3f-4242-90df-f930fe936ac0)

Tracking 된 객체가 frame마다 얼마나 이동하는지를 계산하여 움직이는 객체, 움직이지 않는 객체를 구분하였습니다. 

각 `frame, class name(ex, car, person)`별로 탐지된 객체들의 `bounding box의 좌측상단(혹은 우측상단) point의 변동`을 계산하여 평균을 계산합니다. 

다음 frame에서 탐지된 객체의 `bounding box 좌측상단 point의 변동 * 가중치(hyper parameter)`의 값이 이전 frame에서 계산된 bounding box들의 변동의 평균값보다 더 크다면 움직이는 객체(Moving)로 판단합니다. 


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
                    #colors[track_id % len(colors)], 
                    (0, 0, 255), 
                    thickness=2)
        
        print(f'A moving {class_name} detected!!!')

<br/>
<br/>
<br/>

### **2. Moving 상태가 지정한 조건만큼 연속해서 지속될 시 'Warning'상태가 됩니다.**

WARNING을 부여받은 객체는 알고리즘이 예의주시 하고 있다는 뜻입니다 👀. 




    # WARNING을 주는 기준 : warning_count
    # warning 조건 달성
    if globals()[f'track_id_{track_id}_count_{sequence}'] >= warning_count:
        cv2.putText(frame, "WARNING", 
                    (int(x1), int(y1) - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 165, 255), thickness=4)

<br/>
<br/>
<br/>

### **3. False Alarm을 예방하고자 탐지하고자 하는 구간(Zone)을 설정해줍니다.**  
![image](https://github.com/CodeofO/2023_06_V2_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/1bf4009b-4d53-43eb-9d3b-ade207f0263b)
    
    # 두 영상에 모두 WARNING 신호를 줄 때
    if (warning_text1_source != '') & (warning_text2_source != '') & ((count_in_zone_1 > 0) & (count_in_zone_2 > 0)): 
        c += 1
        warning_text1 = warning_text1_source
        warning_text2 = warning_text2_source

**LED 점등 조건**  
조건1. 두 영상에 모두 Warning 상태인 객체가 있을 때    
조건2. Warning 상태인 두 객체가 Zone 안에 있을 때    
  
<br/>
<br/>
<br/>
        
✅ **조감도**  
![image](https://github.com/CodeofO/2023_06_V2_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/52e4acbe-c624-4221-891b-863aadad77c9)

<br/>
<br/>
<br/>
  
✅ **최종 결과물** 
<img width="1361" alt="image" src="https://github.com/CodeofO/2023_06_V2_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/f9890d6c-16ad-4cf9-ba4c-f0cf58ee8412">

  
