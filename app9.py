import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 무엇인가요?",
    "모델은 어떤걸 사용했나요?",
    "데이터는 어떻게 구하였나요?",
    "몇주동안 프로젝트를 진행하셨나요?",
    "조장은 누구인가요?",
    "조원은 누구인가요?",
    "프로젝트 하는데 어려움은 없었나요?"
]

answers = [
    "yolo 를 이용해서 피트니스 자세를 교정해주는 시스템을 만드는 것입니다.",
    "yolo 모델 8 버전을 썼습니다.",
    "ai허브에서 구하였습니다.",
    "총 3주 동안 진행하였습니다.",
    "서동현 입니다.",
    "경진우, 노승욱, 이원석 입니다.",
    "모든게 다 어렵습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포트폴리오 챗봇")

# 이미지 표시
st.image("hand.png", caption="Welcome to the Restaurant Chatbot", use_column_width=True)

st.write("프로젝트에 관한 질문을 입력해보세요. 예: 주제가 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
