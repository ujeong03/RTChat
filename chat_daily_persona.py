import pandas as pd

# 데이터 불러오기 (예: CSV 파일)
df = pd.read_csv('./LocalAIVoiceChat/dataset_diary/preproc_4079_diary.csv')

# 페르소나 클래스 정의
class Persona:
    def __init__(self, author_id, title, sentence, age):
        self.author_id = author_id
        self.title = title
        self.sentence = sentence
        self.age = age

    def profile(self):
        return f"{self.age}세, '{self.title}'라는 일기 제목을 가진 사람"

# 페르소나 리스트 생성
personas = [Persona(row['author_id'], row['title'], row['sentence'], row['author_age']) for _, row in df.iterrows()]

def diary_to_dialogue(persona):
    system_prompts = [
        "요즘 어떻게 지내세요?",
        "지금까지 살아오시면서 후회되는 점이 있으신가요?",
        "젊었을 때와 지금을 비교하면 어떤 점이 달라졌다고 느끼세요?",
        "앞으로의 삶에 대한 생각은 어떠세요?"
    ]
    
    dialogue = []
    dialogue.append(f"👤 {persona.profile()} 님과의 대화")
    dialogue.append(f"🟢 시스템: {system_prompts[0]}")
    dialogue.append(f"🔵 {persona.author_id}: {persona.sentence[:300]}...")  # 일기 내용을 일부 잘라서 대화화
    
    for prompt in system_prompts[1:]:
        dialogue.append(f"🟢 시스템: {prompt}")
        dialogue.append(f"🔵 {persona.author_id}: ({persona.age}세의 시선에서 자연스럽게 이어지는 대답 생성)")  # 여긴 LLM으로 생성
    
    return dialogue


all_dialogues = []

for persona in personas:
    dialogue = diary_to_dialogue(persona)
    all_dialogues.append("\n".join(dialogue))

# 결과 저장 (예: 텍스트 파일)
with open('persona_dialogues.txt', 'w', encoding='utf-8') as f:
    for dialogue in all_dialogues:
        f.write(dialogue + "\n\n" + "-"*50 + "\n\n")

        import openai

openai.api_key = 'YOUR_API_KEY'

def generate_response(prompt, persona_sentence, persona_age):
    system_prompt = f"너는 {persona_age}세 한국인처럼 대답해야 해. 다음 회상 일기를 참고해서 자연스럽게 대답해줘:\n\n{persona_sentence}\n\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()
