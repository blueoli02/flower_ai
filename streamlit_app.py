#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1adSa474torlUsGcI1hjNZFwGM705iJts'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(1):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(1):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(1):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/7QhJFgK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=S4LlJBZ0H0w"
        ],
        'texts': [
            "개나리의 꽃말은 희망과 기대입니다. 봄을 알리는 대표적인 꽃입니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/K0qCj3L/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=nvO5HVXGrDQ"
        ],
        'texts': [
            "다닥냉이의 꽃말은 순수한 마음입니다. 작고 귀여운 들꽃입니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/mGwR9kn/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=Ueq8f1er7Ns"
        ],
        'texts': [
            "모란의 꽃말은 부귀와 영화입니다. 화려하고 우아한 꽃으로 유명합니다."
        ]
    },
    labels[3]: {
        'images': [
            "https://i.ibb.co/8MtdvsK/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=soWRF9NIJp4"
        ],
        'texts': [
            "목련의 꽃말은 고귀함과 자연애입니다. 고상한 아름다움을 상징합니다."
        ]
    },
    labels[4]: {
        'images': [
            "https://i.ibb.co/8XkhQRx/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=_WoNEis4HoY"
        ],
        'texts': [
            "민들레의 꽃말은 행복과 순진함입니다. 길가에서 흔히 볼 수 있는 꽃입니다."
        ]
    },
    labels[5]: {
        'images': [
            "https://i.ibb.co/P1qZ0kV/image.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=M_o7GSDVk-Q"
        ],
        'texts': [
            "벚나무의 꽃말은 아름다운 정신입니다. 봄을 대표하는 나무입니다."
        ]
    },
    labels[6]: {
        'images': [
            "https://i.ibb.co/8stmjZh/1am-NA50dk-GCkn-RBC57i-TGPPYBTh-AOcy-Xxr-VYbd0-Fummq-XZ6g2io-Snn5-GII-vs-U-8-Uqs6-Ddn0-TU1-Qk-J6ra-S.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=aSD_cgRNgus"
        ],
        'texts': [
            "복수초의 꽃말은 영원한 행복과 슬픈 추억입니다. 봄의 전령으로 알려져 있습니다."
        ]
    },
    labels[7]: {
        'images': [
            "https://i.ibb.co/BPCDBzf/20100218054123-1.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=qlecJHm6itk"
        ],
        'texts': [
            "산철쭉의 꽃말은 사랑의 기쁨입니다. 산에서 흔히 볼 수 있는 꽃입니다."
        ]
    },
    labels[8]: {
        'images': [
            "https://i.ibb.co/rQZbpmg/image.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=qBdvCnTuTEk"
        ],
        'texts': [
            "영산홍의 꽃말은 희생과 사랑입니다. 진달래와 유사한 꽃입니다."
        ]
    },
    labels[9]: {
        'images': [
            "https://i.ibb.co/0DvZygX/image.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3LhbpE0mhsk"
        ],
        'texts': [
            "유채의 꽃말은 쾌활함입니다. 노란 꽃으로 널리 알려져 있습니다."
        ]
    },
    labels[10]: {
        'images': [
            "https://i.ibb.co/5YDC0k4/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=j_e-5qpCn3I"
        ],
        'texts': [
            "자목련의 꽃말은 고귀함과 숭고함입니다. 보라빛의 목련입니다."
        ]
    },
    labels[11]: {
        'images': [
            "https://i.ibb.co/TPgfYD2/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=sRn8B7lWAp0"
        ],
        'texts': [
            "패랭이꽃의 꽃말은 순결과 우정입니다. 화분에 키우기 좋은 꽃입니다."
        ]
    },
    labels[12]: {
        'images': [
            "https://i.ibb.co/kGv4b40/image.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=4_ObwHMIPJU"
        ],
        'texts': [
            "팬지의 꽃말은 생각과 사색입니다. 다양한 색상으로 피어납니다."
        ]
    }
}

    

    
# 레이아웃 설정
left_column, right_column = st.columns([5, 4])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 1,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 1,
            'texts': ["기본 텍스트"] * 1
        })
        display_right_content(prediction, data)

