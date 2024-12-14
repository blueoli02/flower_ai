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
            "https://i.ibb.co/8stmjZh/1am-NA50dk-GCkn-RBC57i-TGPPYBTh-AOcy-Xxr-VYbd0-Fummq-XZ6g2io-Snn5-GII-vs-U-8-Uqs6-Ddn0-TU1-Qk-J6ra-S.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=aSD_cgRNgus"
        ],
        'texts': [
            "복수초 - 꽃말: 영원한 행복 또는 슬픈 추억, 한국의 산야에서 봄을 알리는 꽃입니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/BPCDBzf/20100218054123-1.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=qlecJHm6itk"
        ],
        'texts': [
            "산철쭉 - 꽃말: 사랑의 기쁨, 한국의 산림에서 쉽게 볼 수 있는 봄꽃입니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "개나리 - 꽃말: 희망, 희망찬 봄을 알리는 꽃으로 한국에서 흔히 볼 수 있는 관목입니다."
        ]
    },
    labels[3]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "벚나무 - 꽃말: 정신의 아름다움, 봄철에 아름다운 벚꽃을 피우며 일본과 한국에서 상징적인 존재입니다."
        ]
    },
    labels[4]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "민들레 - 꽃말: 행복, 작고 노란 꽃으로 어린 시절의 추억과 소망을 상징합니다."
        ]
    },
    labels[5]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "목련 - 꽃말: 고귀함, 고급스러운 큰 꽃으로 봄의 시작을 알리는 나무입니다."
        ]
    },
    labels[6]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "자목련 - 꽃말: 사랑의 기쁨, 짙은 자주색의 우아한 꽃이 특징입니다."
        ]
    },
    labels[7]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "유채 - 꽃말: 쾌활, 밝은 노란색 꽃으로 봄의 들판을 물들입니다."
        ]
    },
    labels[8]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "영산홍 - 꽃말: 첫사랑, 화려한 붉은 빛의 철쭉류 꽃입니다."
        ]
    },
    labels[9]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "모란 - 꽃말: 부귀, 크고 화려한 꽃으로 부유함과 행복을 상징합니다."
        ]
    },
    labels[10]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "패랭이꽃 - 꽃말: 순결, 섬세하고 단아한 아름다움의 상징입니다."
        ]
    },
    labels[11]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "다닥냉이 - 꽃말: 극복, 작은 크기에도 불구하고 강인한 생명력을 보여줍니다."
        ]
    },
    labels[12]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "팬지 - 꽃말: 사색, 다양한 색상으로 사람들의 마음을 사로잡는 아름다운 꽃입니다."
        ]
    }
}
    

    
# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

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
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

