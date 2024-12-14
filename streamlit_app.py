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
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "복수초는 영원한 행복 또는 슬픈 추억 꽃말을 가지고 있어요.\n은 미나리아재비과에 속하는 복수초는 측금잔화 (황금색 잔모양의 꽃), 원일초(설날에 피는 꽃), 설련화(눈색이꽃, 눈 속에서 꽃이 핀다), 얼음새꽃(빙리화, 얼음 상이에 꽃이 핀다) 등의 다양한 이름으로도 불려요.\n복수초는 줄기가 분지하지 않기 때문에 꽃이 1개만 달리며, 잎보다 꽃이 먼저 개화하고, 꽃잎보다 긴 꽃받침이 8장인 특징을 가지며, 개복수초와 세복수초는 줄기가 분지하고 각각 줄기 끝에 꽃이 달려 꽃이 많이 달리고, 꽃잎보다 짧은 꽃받침을 가진 특징으로 구분됩니다. 또한, 개복수초는 세복수초와 비교하면 줄기의 속이 차있고, 잎의 소열편이 피침형이며, 잎의 선단부는 뾰족하고, 꽃받침 너비가 꽃잎보다 넓으며, 수과괴는 타원형인 특징으로 구분됩니다. 국내에서 생육하는 복수초는 해발 800m의 높은 산지에 생육하고 있어서 주변에서 흔히 보이는 복수초속 식물은 대부분 개복수초이며, 특히 제주도에서는 세복수초만이 분포하고 있습니다. "
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
          
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
       
        ],
        'texts': [
            "봇치더락입니다."
     
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
            "스즈매의 문단속"
        
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
            "장송의프리렌"
         
        ]
    }
}
    labels[0]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "명탐정코난"
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
          
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
       
        ],
        'texts': [
            "봇치더락입니다."
     
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
            "스즈매의 문단속"
        
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
            "장송의프리렌"
         
        ]
    }
}
    labels[0]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "명탐정코난"
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
          
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
       
        ],
        'texts': [
            "봇치더락입니다."
     
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
            "스즈매의 문단속"
        
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
            "장송의프리렌"
         
        ]
    }
}
    labels[0]: {
        'images': [
            "https://via.placeholder.com/300?text=Label1_Image1"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "명탐정코난"
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1"
          
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
       
        ],
        'texts': [
            "봇치더락입니다."
     
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
            "스즈매의 문단속"
        
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
            "장송의프리렌"
         
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

