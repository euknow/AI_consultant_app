from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import openai

openai.api_key = "your api key"

def chat(messages):        
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    resp_dict = response.to_dict_recursive() # 깊은 구조의 openai return 풀어서 dictionary로 리턴
    assistant_turn = resp_dict['choices'][0]['message']
    
    return assistant_turn # {"role": "assistant", "content": "blahblahblah"}

app = FastAPI()

class Turn(BaseModel):
    role: str
    content: str

class Messages(BaseModel):
    messages: List[Turn]  # [{"role": "user", "content": "blahblahblah"}, {"role": "assistant", "content": "blahblahblah"}, ...]


@app.post("/chat", response_model=Turn)



# def post_chat(messages: Messages):
#     system_instruction = "너의 역할은 '어린이 심리 상담사'야. 진로, 심리, 관심사, 가정 내 불화, 재능이 있는 분야 등에 대해 한 번에 하나씩 질문하고 대답을 받아보도록 해줘."
#     messages = messages.dict()
#     messages['messages'].insert(0, {"role": "system", "content": system_instruction})
#     assistant_turn = chat(messages=messages['messages'])
#     return assistant_turn

def post_chat(messages: Messages):
    system_instruction = """너의 이름은 'AI상담가 해피'야. 
                            너의 역할은 어린이와 친구처럼 지내면서 어린이와 비슷한 말투와 어휘로 상담을 해주는 역할이야. 
                            너의 user는 어린이면서 너의 상담 대상이야. 
                            user의 이름을 알게되면 자연스럽게 질문에 이름을 포함해줘 
                            너는 user에게 존댓말이 아닌 낮춤말로 질문을 해야 해.
                            너는 user에게 먼저 밑의 번호로 기재된 질문들을 순서대로 하나씩 하고 user의 대답을 들어주고 공감하는 방식으로 상담을 진행해줘.
                            0. 안녕, 너의 이름은 머야?
                            1. 요즘 잘지내고 있어? 기분은 어때?
                            2. 요즘 가장 재밌어하는 게 뭐야?
                            3. 요즘 어린이집에서 무엇을 하고 있어? 재밌는 건 어떤 거였어?
                            4. 너가 가장 좋아하는 놀이나 운동이 뭐야?? 아니면 게임 좋아해?
                            5. 밥은 잘 먹고 있어? 어떤 반찬이 가장 맛있어?
                            6. 유치원에서 가장 친한 친구 이름이 뭐야? 친구들이랑 무얼 하면서 놀아?
                            7. 선생님은 이야기를 잘 들어줘? 혹시나 서운한 점은 있어?
                            8. 집에서 부모님이랑 잘 지내?? 부모님이랑 갔던 곳 중에서 재밌었던 곳이 어디야?
                            9. 오늘 대화는 여기까지야. 언제든지 내가 필요하면 찾아와줘. 다음에 또 보자.
            
                            2문장 이내로 짧게 답변하면 좋겠어.
                            """
                            
    messages = messages.dict()
    messages['messages'].insert(0, {"role": "system", "content": system_instruction})
    
    assistant_turn = chat(messages=messages['messages'])
    
    return assistant_turn

# fastapi에서 업로드된 파일을 받는 방법
@app.post("/transcribe")
def transcribe_audio(audio_file : UploadFile = File(...)):
    
    try:
    
        file_name = "tmp_audio_file.wav"
        with open(file_name, "wb") as f :
            f.write(audio_file.file.read())
            
        with open(file_name, "rb") as f:
            transcription = openai.Audio.transcribe("whisper-1", f)
            
        text = transcription["text"]
    except Exception as e : 
        print(e)
        text = f"음성 인식에 실패했습니다. {e}"
    
    return {"text" : text}
