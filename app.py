import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import altair as alt
import streamlit as st

import io
from streamlit_chat import message
import requests

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json


# -------------------- ▼ 필요 변수 생성 코딩 Start ▼ --------------------

# 전체 유치원생 데이터
member = pd.read_csv('./member.csv', encoding='cp949')
EMOTIONS = {0:'Angry', 1 :'Happy', 2: 'Neutral', 3:'Sad', 4: 'Surprise'}

# -------------------- ▼ 함수 ▼ -----------------------------

# 이름 이쁘게 만들기
def char_sex(info):
    name, sex = info
    if sex =="남":
        return f"👦🏻 {name}"
    else :
        return f"👧🏻 {name}"

def chat(text):
    if text != st.session_state["before_text"] :
        user_turn = {"role":"user", "content":text}
        messages = st.session_state["messages"]
        resp = requests.post(chat_url, json={"messages":messages+[user_turn]})
        assistant_turn = resp.json()
    
        st.session_state["before_text"] = text
        st.session_state["messages"].append(user_turn)
        st.session_state["messages"].append(assistant_turn)

        return assistant_turn["content"]

def sum_chat(text):
    user_turn = {"role":"user", "content":text}
    messages = st.session_state["messages"]
    resp = requests.post(chat_url, json={"messages":messages+[user_turn]})
    assistant_turn = resp.json()

    return assistant_turn["content"]

def emotion():
    
    pred = None  # pred 변수 초기화
    
    # Load Model
    json_file = open('./model/emotion_model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_classifier = model_from_json(loaded_model_json)
    emotion_classifier.load_weights("./model/emotion_model1.h5")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
    
    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Set the video capture resolution
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    # 웹캠 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 얼굴 감정 분석 및 결과 표시
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            pred = emotion_classifier.predict(face_roi, verbose = 0)[0]
            
            if pred is not None:
                emotion_probability = np.max(pred)
                label = list(EMOTIONS.keys())[np.argmax(pred)]

                cv2.putText(frame, EMOTIONS[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (135, 206, 235), 2)
                
                st.session_state["stacked_array"] = np.vstack((st.session_state["stacked_array"], pred))
                fig = radar_chart(pred)
                df = pd.DataFrame(st.session_state['stacked_array']).rename(
                    columns={0:'화남', 1:'기쁨', 2:'중립', 3:'슬픔', 4:'당황'})
        
            st.session_state["webcam_placeholder"].image(frame, channels="RGB", use_column_width='auto')
            st.session_state["radar_placeholder"].plotly_chart(fig, use_container_width=True)
            st.session_state["area_placeholder"].area_chart(df, use_container_width=True, height=470)
                
        # 중지 버튼 클릭 시 웹캠 루프 중단
        if st.session_state["stop"] == 1 :
            break
    
    cap.release()
    cv2.destroyAllWindows()

# plotly 레이더 차트 생성
def radar_chart(pred):
    
    # plotly 레이더 차트 생성
    radar = pd.DataFrame(dict(
        r = pred,
        theta = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    ))

    st.session_state["radar_chart"] = px.line_polar(radar, r='r', theta=radar['theta'].astype(str), 
                        line_close=True, # 선 연결
                        template = 'plotly_white')
                
        
    st.session_state["radar_chart"].update_layout(

        autosize = True,  # 크기 자동조정

        # 극좌표 설정
        polar=dict(
            # 반지름 축 설정
            radialaxis=dict(
                angle=45,            # 반지름 축 각도 (기본값 0) 
                autorange = False,   # 자동범위 (기본값 True, 난 설정할거라 False로)
                visible=True,        # 전부 보임
                showline = False,    # 선 안보이게, 텍스트만 보이게
                gridcolor = 'lightgray',
                tickmode = 'array',
                tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],              # 간격 값
                ticktext = ['0', '0.2', '0.4', '0.6', '0.8', '1'],  # 간격 라벨
                tickangle = 45                                      # 라벨 회전
            ),
            # 각도 축 설정
            angularaxis=dict(
                rotation=90,
                gridcolor = 'lightgray',
                linecolor = 'black',
                direction = 'clockwise',
            )
        ),
        font=dict(
            color='black',
            size=15
        ),
    )

    st.session_state["radar_chart"].update_traces(
        fill='toself',
        fillcolor='lightskyblue',
        opacity=0.3,
        line=dict(color='darkblue')
    )
    
    return st.session_state["radar_chart"]

# ===========================================================================

def main():
    
    col_10, col_11 = st.columns([0.3, 0.7])
        
    with col_10:

        with st.container():
            st.info('웹캠')
            # 웹캠 공간생성
            st.session_state["webcam_placeholder"] = st.empty()
            
            st.info('레이더 차트')
            # radar 차트 공간생성
            st.session_state["radar_placeholder"] = st.empty()

    with col_11:
        with st.container():
            st.info('area 차트')
            # area 차트 공간생성
            st.session_state["area_placeholder"] = st.empty()
                
    # 반복문 실행
    emotion()

# -------------------- ▼ session_state ▼ -----------------------

# sidebar에서 선택된 학생
if "selected_name" not in st.session_state:
    st.session_state["selected_name"] = ""

# 유치원생 개별정보
if "preschooler" not in st.session_state:
    st.session_state["preschooler"] = ""
    
# 슬라이드에 사용된 최소 날짜    
if "min_date" not in st.session_state:
    st.session_state["min_date"] = ""
    
# 슬라이드에 사용된 최대 날짜    
if "max_date" not in st.session_state:
    st.session_state["max_date"] = ""   
    
# 유치원생 일지별 감정정보
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""
         
# 기간 변화에 따른 데이터프레임 공유
if "df_emotion" not in st.session_state:
    st.session_state["df_emotion"] = ""    
    
if "before_text" not in st.session_state:
    st.session_state["before_text"] = ""
    
# 감정비율 누적 - area chart 용도
if "stacked_array" not in st.session_state:
    st.session_state["stacked_array"] = np.empty((0, 5))

# 웹캠 공간
if "webcam_placeholder" not in st.session_state:
    st.session_state["webcam_placeholder"] = ''
    
# 레이더 차트 공간
if "radar_placeholder" not in st.session_state:
    st.session_state["radar_placeholder"] = ''

# area 차트 공간
if "area_placeholder" not in st.session_state:
    st.session_state["area_placeholder"] = ''
    
# -------------------- ▼ WEB INTERFACE ▼ -----------------------    
    
# 레이아웃 구성하기 
st.set_page_config(
    page_icon="❤️",
    page_title="아이 am happy!",
    layout="wide")

name_list = member[["NAME", "SEX"]]
name_list = list(name_list.apply(char_sex, axis=1))

with st.sidebar:
    # st.sidebar.image("./image/kindergarten.png", width=100)
    st.markdown('<h1 style="font-size: 24px; font-weight: bold;">🧚원생을 선택하세요</h1>', 
                unsafe_allow_html=True)    
    
    selected_name = st.selectbox('', name_list, index=0, key="selected_one")
    selected_name = selected_name.split(" ")[-1]    
    
    st.session_state["selected_name"] = selected_name
    
    st.session_state["preschooler"] = member.loc[member['NAME'] == st.session_state["selected_name"], :]

if st.session_state["selected_name"] is not None:
    
    # 유치원생 일지별 감정정보 읽어오기
    df = st.session_state['preschooler']
    st.session_state["emotion"] = pd.read_csv(
        f"./member/{st.session_state['preschooler'].loc[:, 'ID'].item()}{st.session_state['selected_name']}.csv", encoding = 'cp949')
    st.session_state["child_image"] = f"./child_image/{st.session_state['preschooler'].loc[:, 'ID'].item()}{st.session_state['selected_name']}.png"
    
    tab1, tab2, tab3 = st.tabs(["영유아 정보", "AI 상담", "감정 인식"])
    
# -------------------------- tab1 ------------------------------

    with tab1:
        with st.expander(" ", expanded=True):
            st.image("./image/kindergarten.png", width=1850)        
        
        col_00, col_01 = st.columns([1, 4])
        
        with col_00:
            st.info('😊원생 정보😊')
            st.image(st.session_state["child_image"], width = 350)
            
            for idx, col in enumerate(['ID', '이름', '성별', '생년월일', '나이', '키', '체중', '혈액형', '주소']):                     
                col_00_00, col_00_01 = st.columns([1,2])
                with col_00_00:
                    st.info(f"{col}")

                with col_00_01:
                    st.text_input(label=f"{col}",
                                  value=f"{st.session_state['preschooler'].iloc[:, idx].item()}", label_visibility="collapsed")
        
        with col_01:
            st.success('등원 시 일별 감정 통계')

            # 데이터프레임 생성
            df_emotion = st.session_state["emotion"].groupby(
                by='Emotion', as_index=False)['Emotion'].count()
            df_emotion = df_emotion.rename(columns={'Emotion': 'count'})
            df_emotion['Emotion'] = ['화남', '기쁨', '무표정', '슬픔', '당황']

            # 슬라이더
            min_date = datetime.datetime.strptime(st.session_state["emotion"]['Date'].min(), "%Y-%m-%d")
            max_date = datetime.datetime.strptime(st.session_state["emotion"]['Date'].max(), "%Y-%m-%d")      

            slider_date = st.slider('날짜', min_value = min_date, max_value = max_date,
                                    value=(min_date, max_date))
            
            st.session_state["min_date"] = slider_date[0]
            st.session_state["max_date"] = slider_date[1] 
            
            col_01_00, col_01_01 = st.columns([1.5,2.5])
            
            with col_01_00:
                # 파이 차트 생성
                df = st.session_state["emotion"]
                df["Date"] = pd.to_datetime(df["Date"])
                
                # st.session_state["emotion"]['Date'] = pd.to_datetime(st.session_state["emotion"]['Date'])
                df_emotion = df[(slider_date[0] <= st.session_state["emotion"]['Date']) \
                                & (slider_date[1] >= st.session_state["emotion"]['Date'])]
                
                df_emotion = df_emotion.groupby(by='Emotion', as_index=False)['Emotion'].count()
                df_emotion = df_emotion.rename(columns={'Emotion': 'count'})
                df_emotion['Emotion'] = ['화남', '기쁨', '무표정', '슬픔', '당황']
                df_emotion = df_emotion.sort_values('Emotion', ascending=True)
                st.session_state["df_emotion"] = df_emotion
                
                fig = px.pie(df_emotion, values='count', names='Emotion', height=450, width=580, hole=0.3,
                             color_discrete_sequence=['orange', 'red', 'blue', 'purple', 'green'])

                fig.update_traces(textposition='inside', textinfo='percent+label')
                #fig.update_layout(title='3개월 감정 비율')
                
                # 차트 출력
                st.plotly_chart(fig)
            
            with col_01_01:            
                # Display bar chart with custom colors
                
                fig = px.bar(df_emotion, x='Emotion', y='count',text_auto=True) # text_auto=True 값 표시 여부
                
                # Customize the color scale
                colors = ['orange', 'purple', 'green', 'blue', 'red']
                fig.update_traces(marker_color=colors)

                # Set the chart size
                # fig.update_layout(height=400, width=800)

                # Display the chart in Streamlit             
                st.plotly_chart(fig, use_container_width=True,
                                width=0, height=250)                
                
            child_name = st.session_state["selected_name"]
            child_min_date = st.session_state["min_date"].strftime("%Y-%m-%d")
            child_max_date = st.session_state["max_date"].strftime("%Y-%m-%d")
            child_emotion = st.session_state["df_emotion"]
            child_max_emotion = child_emotion.sort_values(by="count",ascending=False).iloc[0,1]
            st.info(f"{child_min_date}부터 {child_max_date}까지 {child_name}는(은) 대체적으로 기분이 {child_max_emotion}인 것 같습니다.")
            
            st.warning("😊주별 상담을 통한 행복지수 변화")
            df = st.session_state["emotion"]
            df["Date"] = pd.to_datetime(df["Date"])
            emotion_df  = df[(slider_date[0] <= st.session_state["emotion"]['Date']) \
                            & (slider_date[1] >= st.session_state["emotion"]['Date'])]       
                            
            # chart = alt.Chart(emotion_df).mark_line(color='blue').encode(x='Date', y='Happiness_ratio').properties(width=325, height=400)
            # st.altair_chart(chart, use_container_width=True)    

            line_placeholder = st.empty()
            df = st.session_state["emotion"].loc[:, ['Week', 'Happiness_ratio']].drop_duplicates().reset_index(drop=True)
            fig = px.line(data_frame=df, x='Week', y='Happiness_ratio', 
                          text = 'Happiness_ratio', 
                          labels = {'Week':'주차', 'Happiness_ratio': '행복지수'},
                          range_y = (0, 6), template = 'plotly_white'
                          )
            fig.update_traces(textposition="top center")
            line_placeholder.plotly_chart(fig, use_container_width=True)
            
# ---------------------- tab2 상담하기 -----------------------------      
    with tab2:
        host_url = "http://localhost:8000"
        chat_url = f"{host_url}/chat"
        transcribe_url = f"{host_url}/transcribe"
        
        if "messages" not in st.session_state:
            st.session_state['messages'] = []
        
        col_00, col_01, col_02 = st.columns([2,1,1])
        
        with col_00 :
            st.success("🦾믿음직한 우리 아이 상담 AI : 해피")
            col_00_00, col_00_01 = st.columns([1,3])
            
            with col_00_00 :
                st.image("./image/happy.png", width=250)

        
            with col_00_01 :
                
                row1 = st.container()
                row2 = st.container()
                
                with row2:
                    input_text = st.text_input("You")
                    if input_text:
                        chat(input_text)                   
                        
                with row1:
                    for i, msg_obj in enumerate(st.session_state['messages']):
                        msg = msg_obj["content"]
                        
                        is_user = False
                        if i % 2 == 0:
                            is_user = True
                        
                        message(msg, is_user=is_user, key=f"chat_{i}") #key를 자동으로 증가하는 방식으로 입력함으로써 중복을 방지
                 
        with col_01 :
            st.info("💬대화 내용 요약")
            finish_button = st.button("요약", key="finish_button", use_container_width=True)
            if finish_button : 
                sum_chat_output = sum_chat("지금까지 대화 한 내용을 상담 대상의 학부모가 보기 편하게 3줄로 요약해줘")
                st.info(sum_chat_output)     
                
        with col_02 :                
            st.warning("👨‍🏫맞춤 가이드")
            custom_guide = st.button("보기", key="custom_guide", use_container_width=True)
            if custom_guide:
                guide_output = sum_chat("지금까지 대화 한 내용을 기반으로 상담 대상의 학부모에게 맞춤 육아 가이드를 제공해줘.")
                st.warning(guide_output)     
                
# ---------------------- tab3 감정 인식 -----------------------------      
    with tab3:
        st.header("웹캠 얼굴 감정 인식")

        col_00, col_01, col_02 = st.columns([0.35, 0.2, 0.35])
        with col_00:
            pass
        with col_01:
            st.session_state["start"] = st.button("시작", use_container_width = True)
            st.session_state["stop"] = st.button('종료', use_container_width = True)
        with col_02:
            pass

        if st.session_state["start"] == 1:

            # main 함수 호출
            main()
