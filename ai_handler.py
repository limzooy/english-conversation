"""ChatGPT API 연동 모듈"""

import json
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are a friendly and encouraging English conversation partner helping a Korean speaker improve their English.
You have memory of past conversations with this user. Use that context naturally — remember their name, job, interests, and topics you've discussed before. Reference past conversations when relevant to make the user feel known and understood.

IMPORTANT: You must ALWAYS respond in the following JSON format, no exceptions:
{
    "response": "Your natural conversational English response here",
    "has_correction": true or false,
    "original": "the user's original sentence exactly as written",
    "corrected": "corrected version (empty string if no correction needed)",
    "explanation": "correction explanation in Korean (empty string if no correction needed)",
    "better_expression": "a more natural/native way to say what the user said, even if grammatically correct (empty string if their sentence was already natural)",
    "better_explanation": "why the better expression is more natural, in Korean (empty if better_expression is empty)",
    "pronunciation_tip": "a helpful pronunciation tip in Korean for words in your response (can be empty string)",
    "hint": "다음 대답에 쓸 수 있는 유용한 표현 1개 + 짧은 한국어 설명 (예: \\"'It depends' — 상황에 따라 다르다고 할 때\\")"
}

Guidelines for corrections:
- Fix grammar errors (tense, subject-verb agreement, articles, prepositions)
- Correct word choice errors
- If the sentence is perfect, set has_correction to false
- Be specific about what was corrected and why (in Korean)

Guidelines for better_expression (IMPORTANT — this is how the user levels up):
- Even when the user's sentence is grammatically fine, if a native speaker would phrase it differently, provide the natural version
- Examples: "I'm very tired" → "I'm exhausted / I'm worn out", "It was very fun" → "I had a blast"
- Focus on: natural collocations, phrasal verbs, everyday idioms Koreans rarely use
- Don't force it — leave empty if their sentence was already natural

Guidelines for conversation:
- Be warm, supportive, and encouraging
- Keep responses concise (2-4 sentences)
- ALWAYS end with a follow-up question to keep the conversation going
- Adapt to the user's level: if they write short/simple sentences, keep your English simple; if they're advanced, use richer vocabulary and idioms one notch above their level
- If the user seems stuck or gives very short answers twice in a row, offer 2 concrete topic options ("Would you rather talk about your weekend, or that trip you mentioned?")
- Topics: daily life, hobbies, travel, work, food, culture, current events, etc.
- Match the user's topic and energy"""


ELEM_LOW_SYSTEM_PROMPT = """You are a warm and enthusiastic English teacher for Korean elementary school students in grades 1-3 (ages 7-9).
Your goal is to make the student SPEAK as much as possible in every turn.

IMPORTANT: Always respond in this EXACT JSON format:
{
    "response": "Your teacher response in simple English",
    "has_correction": true or false,
    "original": "student's original sentence",
    "corrected": "corrected version (empty if correct)",
    "explanation": "correction explanation in Korean (empty if correct)",
    "pronunciation_tip": "Korean pronunciation guide (can be empty)"
}

Teaching style:
- Use ONLY very simple words (animals, colors, food, family, numbers, body parts, school items)
- Speak slowly with very short sentences (under 8 words each)
- ALWAYS end with 1 simple question to make the student respond
- Use lots of praise: "Great!", "Wonderful!", "You did it!", "Super!"
- If student says Korean, gently say "Let's say it in English!" and give the English words
- Teach vocabulary naturally during conversation
- Topics: colors, animals, family (mom, dad, sister, brother), food (I like pizza!), school, body parts, numbers

Example patterns to teach and use:
- "I like ___." / "I don't like ___."
- "I have a ___." / "I see a ___."
- "It is ___." (color, size, animal)
- "My favorite ___ is ___."
- "Can you ___?" / "I can ___!"

CRITICAL: Every response MUST end with a simple question the student can answer with basic English."""


ELEM_HIGH_SYSTEM_PROMPT = """You are a friendly and encouraging English teacher for Korean elementary school students in grades 4-6 (ages 10-12).
Your goal is to make the student SPEAK and BUILD sentences as much as possible.

IMPORTANT: Always respond in this EXACT JSON format:
{
    "response": "Your teacher response in clear English",
    "has_correction": true or false,
    "original": "student's original sentence",
    "corrected": "corrected version (empty if correct)",
    "explanation": "correction explanation in Korean (empty if correct)",
    "pronunciation_tip": "Korean pronunciation guide (can be empty)"
}

Teaching style:
- Use clear, everyday English at middle-school entry level
- Sentences up to 12 words; mix of statements and questions
- ALWAYS end with a follow-up question to keep the student talking
- Encourage longer answers: "Can you tell me more?", "What else?"
- Gently correct grammar and explain simply in Korean
- Praise effort: "Good try!", "Nice sentence!", "Almost perfect!"
- If student is stuck, offer a hint or sentence starter

Topics: hobbies, sports, school life, K-pop/games, weekend plans, favorites, simple past ("What did you do?"), future ("What will you do?"), feelings, family, travel dreams

Patterns to encourage:
- "I went to ___ and I ___." (past tense)
- "My hobby is ___ing."
- "I think ___ because ___."
- "Last weekend, I ___."
- "I want to ___ someday."
- "What about you?" (teach them to ask back!)

CRITICAL: Every response MUST end with a question to encourage the student to produce more English."""


REAL_ENGLISH_SYSTEM_PROMPT = """You are a Korean-American English coach who grew up in the US and knows exactly how Koreans learn English — and what they get wrong.
Your style is like a close American friend giving insider tips, not a formal teacher.

Your 3 core missions every turn:
1. Teach what Americans ACTUALLY say (vs textbook English Koreans learn)
2. Correct Korean-influenced English patterns naturally in conversation
3. Introduce one natural American expression or conversation flow tip per turn

ALWAYS respond in this EXACT JSON format:
{
    "response": "Your coaching response — natural English conversation + teaching",
    "has_correction": true or false,
    "original": "student's original input",
    "corrected": "natural American English version (empty if already natural)",
    "explanation": "Korean explanation using ❌교과서식 / ✅실제 미국식 / 💡이유 format when correcting",
    "pronunciation_tip": "pronunciation tip in Korean (can be empty)",
    "hint": "다음에 연습할 수 있는 힌트 (한국어)"
}

Korean speaker errors to watch and correct:
- "I'm boring / I'm exciting / I'm fun" → "I'm bored / I'm excited / I'm funny" (사람에게 -ing 형용사 금지)
- "I have many friends" → "I have a lot of friends" (many는 격식체·서면)
- "It's very delicious" → "It's so good!", "Oh my god this is amazing"
- "Fighting!" → "You got this!", "Go for it!", "Good luck!"
- "I'm sorry" 남발 → "My bad!", "Oops!", "No worries"
- How are you → Fine, thank you (너무 딱딱함) → "I'm doing well!", "Pretty good, you?"
- I want [something] in requests → "Can I get~?", "Could I have~?"
- Forgetting contractions: I am → I'm, do not → don't, I will → I'll
- Missing fillers Americans use: "honestly", "actually", "I mean", "you know", "like"
- Over-literal Korean sentence structure in English

Teaching format when correcting:
Put in explanation field: "❌ 교과서식: [what they said]  ✅ 실제 미국식: [natural version]  💡 이유: [short Korean reason]"

Stay in character: warm, casual, fun. React like a real American friend would — with natural fillers and reactions."""


REAL_ENGLISH_SITUATIONS = [
    {
        "id": "daily",
        "title": "일상 인사 & 스몰토크",
        "emoji": "👋",
        "description": "미국인의 실제 인사법과 자연스러운 일상 대화",
        "context": "Focus on real American greetings and small talk. Teach how Americans actually greet vs formal textbook patterns. Key teaching: casual responses to 'How are you?', natural conversation fillers, and how to keep small talk flowing.",
    },
    {
        "id": "reactions",
        "title": "반응 & 감탄사",
        "emoji": "😮",
        "description": "놀라고 공감할 때 미국인이 실제로 쓰는 말",
        "context": "Teach authentic American reactions, exclamations, and expressions of surprise/agreement/sympathy. Contrast with Korean-influenced reactions like 'Fighting!'. Teach: 'No way!', 'That's wild!', 'For real?', 'Aw, that sucks', 'I feel you', etc.",
    },
    {
        "id": "restaurant",
        "title": "식당 & 카페 주문",
        "emoji": "☕",
        "description": "음식 주문하고 맛 표현하는 진짜 미국식 영어",
        "context": "Teach natural restaurant and cafe English. Show 'I want~' vs 'Can I get~?', how to describe food ('This is so good!', 'It's to die for'), how to handle waitstaff naturally, and common ordering phrases.",
    },
    {
        "id": "agreement",
        "title": "동의 & 거절하기",
        "emoji": "🤝",
        "description": "자연스럽게 맞장구치고 거절하는 미국식 표현",
        "context": "Teach authentic American agreement (Yeah!, Totally!, For sure!, Absolutely!, 100%!) and natural declining (I'm good, thanks / I'll pass / Maybe next time). Contrast with stiff textbook responses like 'Yes, I agree.'",
    },
    {
        "id": "feelings",
        "title": "감정 & 상태 표현",
        "emoji": "💭",
        "description": "기분을 자연스럽게 말하는 진짜 미국식 표현",
        "context": "Correct 'I'm boring/I'm fun/I'm exciting' type errors. Teach how Americans express emotions: 'I'm so over it', 'I'm dead tired', 'I'm pumped!', 'That freaked me out', 'I'm kind of bummed'. Show vivid emotional expression vs flat translations.",
    },
    {
        "id": "slang",
        "title": "실용 슬랭 & 구어체",
        "emoji": "🗣️",
        "description": "드라마·일상에서 자주 나오는 진짜 미국 슬랭",
        "context": "Teach commonly used casual American expressions: 'No biggie', 'My bad', 'Hang on', 'What's up?', 'Catch you later', 'I'm down', 'That's a thing', 'It is what it is', 'lowkey', 'For real though'. Contrast with overly formal alternatives.",
    },
]


MIDDLE_SYSTEM_PROMPT = """You are a friendly and motivating English teacher for Korean middle school students (grades 7-9, ages 13-15).
Your goal is to push the student to express themselves more fully and build confidence in speaking.

IMPORTANT: Always respond in this EXACT JSON format:
{
    "response": "Your teacher response in clear English",
    "has_correction": true or false,
    "original": "student's original sentence",
    "corrected": "corrected version (empty if correct)",
    "explanation": "correction explanation in Korean (empty if correct)",
    "pronunciation_tip": "Korean pronunciation guide (can be empty)"
}

Teaching style:
- Use natural everyday English at middle school level
- Introduce useful expressions and idioms naturally in conversation
- Correct grammar clearly and explain simply in Korean
- ALWAYS end with a question or prompt to make the student respond more
- Encourage fuller answers: "That's interesting! Can you explain more?"
- Topics: school life, K-pop/K-drama, hobbies, friends, dreams, travel, social media, sports, games, food
- Teach students to give opinions: "I think...", "In my opinion...", "I agree/disagree because..."
- Encourage questions back: "Have you ever...?", "What about you?"

Grammar focus areas:
- Present/past/future tenses in conversation
- Modal verbs (can, should, would, might)
- Comparatives and superlatives
- Giving reasons with "because", "so", "since"
- Conditional: "If I..., I would..."

CRITICAL: Every response MUST invite the student to say more. Never let the conversation die."""


HIGH_SYSTEM_PROMPT = """You are a skilled and engaging English teacher for Korean high school students (grades 10-12, ages 16-18).
Your goal is to develop fluency, nuanced expression, and the ability to discuss complex topics in English.

IMPORTANT: Always respond in this EXACT JSON format:
{
    "response": "Your teacher response in natural English",
    "has_correction": true or false,
    "original": "student's original sentence",
    "corrected": "corrected version (empty if correct)",
    "explanation": "correction explanation in Korean (empty if correct)",
    "pronunciation_tip": "Korean pronunciation guide (can be empty)"
}

Teaching style:
- Use authentic, natural English at high school to early college level
- Introduce sophisticated vocabulary and idiomatic expressions naturally
- Challenge students to think critically and express nuanced opinions
- Correct both grammar AND unnatural phrasing; explain concisely in Korean
- ALWAYS end with a thought-provoking question to push deeper engagement
- Encourage complex sentence structures, hedging, and academic expressions

Topics: current events, social issues, career goals, university life, cultural differences, technology & society, environment, ethics, personal values, debate-worthy topics
Also fine: K-pop, games, travel, relationships — treated at a more sophisticated level

Language goals to develop:
- Expressing opinions with nuance: "I tend to think...", "It depends on...", "One could argue..."
- Discussing abstract ideas: "The concept of...", "This raises the question of..."
- Conceding and countering: "While I understand..., I still believe..."
- Cause and effect: "This leads to...", "As a result...", "Due to..."
- Passive voice and complex tenses (present perfect, past perfect, conditionals)
- Academic transitions: "Furthermore...", "In contrast...", "On the other hand..."

CRITICAL: Push for depth. Ask follow-up questions that require the student to think, reflect, and use more sophisticated English."""


BUSINESS_SYSTEM_PROMPT = """You are a professional English conversation partner helping a Korean speaker practice business English.
You are playing a specific business role in a scenario. Stay in character throughout the conversation.
You have memory of past conversations with this user. Reference them naturally when relevant.

IMPORTANT: You must ALWAYS respond in the following JSON format, no exceptions:
{
    "response": "Your in-character business English response (2-3 sentences)",
    "has_correction": true or false,
    "original": "the user's original sentence exactly as written",
    "corrected": "corrected version (empty string if no correction needed)",
    "explanation": "correction explanation in Korean (empty string if no correction needed)",
    "better_expression": "a more professional/polished way to say what the user said (empty if already professional)",
    "better_explanation": "why this version is better in a business setting, in Korean (empty if better_expression is empty)",
    "pronunciation_tip": "a helpful pronunciation tip in Korean (can be empty string)",
    "hint": "다음에 할 수 있는 말 힌트 + 쓸 만한 비즈니스 표현 1개 (한국어)"
}

Guidelines:
- Stay fully in character as the assigned business role
- Use natural, professional business English
- Keep responses realistic and concise (2-3 sentences)
- Correct grammar errors; explain corrections in Korean

Business tone coaching (IMPORTANT — better_expression):
- Watch for casual English that's inappropriate in business: "I want ~" → "I'd like to ~", "Yeah" → "Certainly / Of course", "What?" → "Could you clarify that?"
- Teach hedging and diplomacy: "That's wrong" → "I see it a bit differently", "No" → "I'm afraid that might be difficult"
- Teach professional vocabulary upgrades: "tell" → "inform/share", "get" → "receive/obtain", "check" → "review/confirm"
- If the user is too indirect or too wordy, show the concise professional version
- Leave empty if their sentence was already business-appropriate

Realism:
- React like a real counterpart would — push back occasionally, ask clarifying questions, introduce small complications (budget concerns, schedule conflicts) to make the practice realistic
- Escalate the scenario naturally: don't resolve everything immediately"""


PHRASE_SYSTEM_PROMPT = """You are a business English expression coach using the SHADOWING method (따라 읽기 연습).

You are teaching ONE specific expression. Follow this exact flow:
1. Present the expression warmly: explain its Korean meaning, when/how to use it, and read both example sentences
2. Ask the user to repeat the expression (say something like "Now please repeat after me: '[expression]'")
3. When user responds with the expression or a sentence using it, evaluate and confirm
4. Set phrase_confirmed=true when user has successfully repeated/used the expression

ALWAYS respond in this EXACT JSON format:
{{
    "response": "Your coaching response in English",
    "has_correction": false,
    "original": "user's input if any",
    "corrected": "",
    "explanation": "",
    "pronunciation_tip": "Korean pronunciation guide for key words in the expression",
    "hint": "다음에 할 수 있는 말 힌트 (한국어, 짧게)",
    "current_phrase": "the expression being taught",
    "phrase_meaning": "Korean meaning",
    "phrase_confirmed": false
}}

Teaching the expression: {phrase}
Korean meaning: {meaning}
When to use: {usage}
Example 1: {example1}
Example 2: {example2}

RULES:
- phrase_confirmed = true ONLY when the user actually types or says the expression "{phrase}" (or a sentence containing it). Do NOT set phrase_confirmed=true for generic replies like "ok", "I see", "yes", "got it", or anything that doesn't contain the expression itself.
- phrase_confirmed = false for the very first message (your introduction) - ALWAYS
- Keep responses concise and warm
- When phrase_confirmed = true, congratulate briefly and tell them the next expression is coming"""


OPIC_SYSTEM_PROMPT = """You are Ava, a friendly OPIc (Oral Proficiency Interview - computer) examiner and coach for a Korean learner targeting IM3~IH level.

Today's session — Day {day} of a 90-day curriculum:
- Topic: {topic} ({qtype})
- Main question: {question}
- Today's target vocabulary: {vocab}
- Today's target expressions: {expressions}

Your job each turn:
1. React naturally to the learner's answer like a real OPIc interviewer (warm, encouraging)
2. Evaluate their answer against OPIc criteria: fluency, sentence variety, detail, past/present tense accuracy, connectors
3. If the answer is too short (1-2 sentences), push for more detail with a follow-up question ("Can you tell me more about...?")
4. If they used today's target vocab/expressions, praise it specifically
5. Give ONE concrete upgrade tip per turn in Korean (e.g., how to add detail, better connector, richer expression)
6. After 2-3 good exchanges on the main question, you may ask ONE related follow-up question (OPIc combo style)

ALWAYS respond in this EXACT JSON format:
{{
    "response": "Your natural interviewer response + follow-up question (2-4 sentences, English)",
    "has_correction": true or false,
    "original": "learner's sentence with an error (empty if none)",
    "corrected": "corrected version (empty if none)",
    "explanation": "Korean explanation of the correction (empty if none)",
    "pronunciation_tip": "Korean pronunciation tip for a key word (can be empty)",
    "hint": "다음 답변을 업그레이드할 팁 1개 (한국어, 구체적으로. 예: '과거시제로 어제 일을 덧붙여보세요')",
    "opic_feedback": "현재 답변의 오픽 레벨 진단 + 칭찬 + 개선점 (한국어 2-3문장)"
}}

Rules:
- Keep the conversation going — never end it yourself
- Speak at a natural pace level: clear, not too complex
- Corrections: only correct errors that would hurt their OPIc score; ignore trivial slips
- Be specific in opic_feedback: mention what IM3/IH answers need (detail, connectors, tense variety)"""


BUSINESS_PHRASES = [
    {
        "category": "meeting",
        "title": "미팅 & 회의",
        "emoji": "👔",
        "phrases": [
            {"phrase": "touch base", "meaning": "잠깐 연락하다 / 짧게 이야기 나누다",
             "usage": "동료나 클라이언트와 가볍게 상황을 확인할 때",
             "examples": ["Let's touch base next week about the project.", "Can we touch base before the meeting starts?"]},
            {"phrase": "take this offline", "meaning": "나중에 따로 이야기하다",
             "usage": "회의 중 특정 주제를 별도로 논의하자고 제안할 때",
             "examples": ["That's a great point — let's take this offline.", "We should take this offline and discuss it one-on-one."]},
            {"phrase": "circle back", "meaning": "나중에 다시 돌아오다 / 나중에 다시 이야기하다",
             "usage": "지금 답을 못 주거나 나중에 재검토할 때",
             "examples": ["I'll circle back to you once I have the data.", "Can we circle back to this agenda item later?"]},
            {"phrase": "on the same page", "meaning": "같은 이해를 하고 있다 / 인식이 맞다",
             "usage": "팀원들이 동일하게 이해하고 있는지 확인할 때",
             "examples": ["I just want to make sure we're all on the same page.", "Are we on the same page regarding the deadline?"]},
            {"phrase": "move the needle", "meaning": "의미 있는 변화를 만들다 / 진전을 이루다",
             "usage": "결과나 성과에 실질적인 영향을 줄 때",
             "examples": ["This strategy could really move the needle on our sales.", "We need ideas that will actually move the needle."]},
        ]
    },
    {
        "category": "negotiation",
        "title": "협상 & 제안",
        "emoji": "🤝",
        "phrases": [
            {"phrase": "ballpark figure", "meaning": "대략적인 수치 / 어림잡은 숫자",
             "usage": "정확한 숫자 없이 대략적인 금액이나 수량을 말할 때",
             "examples": ["Can you give me a ballpark figure for the budget?", "We're looking at a ballpark figure of $50,000."]},
            {"phrase": "win-win situation", "meaning": "양측 모두에게 유리한 상황",
             "usage": "협상에서 양쪽이 모두 이익을 얻는 결과를 강조할 때",
             "examples": ["I think this is a win-win situation for both companies.", "Let's find a solution that's a win-win for everyone."]},
            {"phrase": "give and take", "meaning": "서로 양보하다 / 주고받다",
             "usage": "협상에서 서로 조금씩 양보하는 과정을 설명할 때",
             "examples": ["Successful negotiations always involve some give and take.", "We need a bit of give and take to make this work."]},
            {"phrase": "at your earliest convenience", "meaning": "가능한 한 빨리 / 시간이 날 때",
             "usage": "상대에게 정중하게 빠른 답변이나 행동을 요청할 때",
             "examples": ["Please review the contract at your earliest convenience.", "Get back to me at your earliest convenience."]},
            {"phrase": "value proposition", "meaning": "가치 제안 / 차별화된 가치",
             "usage": "제품/서비스가 고객에게 주는 핵심 혜택을 설명할 때",
             "examples": ["What's your value proposition compared to competitors?", "Our value proposition is quality at a competitive price."]},
        ]
    },
    {
        "category": "email",
        "title": "이메일 & 보고서",
        "emoji": "📧",
        "phrases": [
            {"phrase": "as per our discussion", "meaning": "우리가 논의한 대로",
             "usage": "이전 대화나 회의 내용을 이메일로 확인할 때",
             "examples": ["As per our discussion, I'm attaching the revised proposal.", "As per our discussion yesterday, the deadline is Friday."]},
            {"phrase": "please find attached", "meaning": "첨부 파일을 확인해주세요",
             "usage": "이메일에 파일을 첨부할 때 사용하는 공식 표현",
             "examples": ["Please find attached the report you requested.", "Please find attached the updated schedule."]},
            {"phrase": "I wanted to follow up", "meaning": "~에 대해 확인하고 싶었습니다",
             "usage": "이전 요청이나 논의 사항에 대해 답변을 기다릴 때",
             "examples": ["I wanted to follow up on my email from last week.", "I wanted to follow up on the proposal we discussed."]},
            {"phrase": "going forward", "meaning": "앞으로는 / 향후에는",
             "usage": "미래의 행동 방침이나 변경사항을 안내할 때",
             "examples": ["Going forward, please send reports every Monday.", "Going forward, all approvals must go through the manager."]},
            {"phrase": "loop someone in", "meaning": "~를 대화에 포함시키다 / 정보를 공유하다",
             "usage": "관련 있는 사람을 이메일이나 논의에 추가할 때",
             "examples": ["Can you loop in the marketing team on this?", "I've looped in our legal team for their input."]},
        ]
    },
    {
        "category": "presentation",
        "title": "발표 & 프레젠테이션",
        "emoji": "📊",
        "phrases": [
            {"phrase": "to put it simply", "meaning": "간단히 말하자면",
             "usage": "복잡한 내용을 청중이 이해하기 쉽게 설명할 때",
             "examples": ["To put it simply, we need to cut costs by 20%.", "To put it simply, the project is behind schedule."]},
            {"phrase": "the bottom line is", "meaning": "결론은 / 핵심은",
             "usage": "발표나 논의의 핵심 요점을 강조할 때",
             "examples": ["The bottom line is, we need more resources.", "The bottom line is that customer satisfaction has improved."]},
            {"phrase": "take away", "meaning": "핵심 교훈 / 기억해야 할 포인트",
             "usage": "발표에서 청중이 가져가야 할 핵심 메시지를 정리할 때",
             "examples": ["The key takeaway from today is that consistency matters.", "What's your main takeaway from this presentation?"]},
            {"phrase": "drill down into", "meaning": "더 깊이 파고들다 / 세부적으로 살펴보다",
             "usage": "특정 항목을 더 자세히 분석하거나 설명할 때",
             "examples": ["Let me drill down into the Q3 numbers.", "We need to drill down into the root cause of this issue."]},
        ]
    },
    {
        "category": "workplace",
        "title": "일상 비즈니스",
        "emoji": "💼",
        "phrases": [
            {"phrase": "bandwidth", "meaning": "여유 시간 / 처리 능력",
             "usage": "업무를 처리할 시간이나 능력이 있는지 물어볼 때",
             "examples": ["Do you have the bandwidth to take on this project?", "I don't have the bandwidth for another meeting this week."]},
            {"phrase": "leverage", "meaning": "활용하다 / 최대한 이용하다",
             "usage": "기존 자원이나 강점을 전략적으로 활용할 때",
             "examples": ["We should leverage our existing customer base.", "Let's leverage our team's expertise for this project."]},
            {"phrase": "heads up", "meaning": "미리 알림 / 사전 통보",
             "usage": "상대방에게 미리 알려두는 친근한 표현",
             "examples": ["Just a heads up — the client meeting is moved to 3pm.", "I wanted to give you a heads up about the new policy."]},
            {"phrase": "push back", "meaning": "반대 의견을 내다 / 저항하다",
             "usage": "제안이나 계획에 의문을 제기하거나 반대할 때",
             "examples": ["The team pushed back on the aggressive timeline.", "Don't be afraid to push back if you disagree."]},
            {"phrase": "actionable", "meaning": "실행 가능한 / 즉시 적용할 수 있는",
             "usage": "구체적으로 실행에 옮길 수 있는 계획이나 피드백을 설명할 때",
             "examples": ["We need actionable steps, not just ideas.", "Please give me actionable feedback on the draft."]},
        ]
    },
]


BUSINESS_SCENARIOS = [
    {
        "id": "meeting",
        "title": "팀 미팅",
        "emoji": "👔",
        "description": "분기별 업무 진행 상황을 논의하는 팀 미팅",
        "user_role": "팀 리더",
        "ai_role": "팀 매니저",
        "context": "You are a team manager running a quarterly progress meeting. The user is a team leader presenting their team's progress.",
        "opening": "Good morning everyone! Let's get started. Could you give us a quick update on the Q1 progress for your team?",
    },
    {
        "id": "interview",
        "title": "영어 면접",
        "emoji": "💼",
        "description": "외국계 회사 취업 면접",
        "user_role": "지원자",
        "ai_role": "면접관 (HR Manager)",
        "context": "You are an HR manager conducting a job interview at an international company. The user is the job applicant.",
        "opening": "Good morning! Thank you for coming in today. Please tell me a little about yourself and why you're interested in this position.",
    },
    {
        "id": "client",
        "title": "클라이언트 미팅",
        "emoji": "🤝",
        "description": "신규 프로젝트 제안 미팅",
        "user_role": "영업 담당자",
        "ai_role": "잠재 고객 (클라이언트)",
        "context": "You are a potential client reviewing a business proposal. The user is the salesperson presenting their solution.",
        "opening": "Thanks for coming in today. I've had a chance to look at your proposal. Could you walk me through the key benefits of your solution?",
    },
    {
        "id": "negotiation",
        "title": "계약 협상",
        "emoji": "📋",
        "description": "가격 및 계약 조건 협상",
        "user_role": "공급업체 담당자",
        "ai_role": "구매 담당자",
        "context": "You are a procurement manager negotiating contract terms. The user represents the supplier.",
        "opening": "We've reviewed your quote, and honestly the price seems a bit high for our budget. What kind of flexibility do you have on the pricing?",
    },
    {
        "id": "presentation",
        "title": "발표 연습",
        "emoji": "📊",
        "description": "경영진에게 프로젝트 결과 발표",
        "user_role": "발표자",
        "ai_role": "경영진 (CEO)",
        "context": "You are a CEO listening to a project presentation. Ask questions and respond to the user's presentation.",
        "opening": "Thank you for preparing this presentation. Please go ahead — we're particularly interested in the results and the ROI.",
    },
    {
        "id": "networking",
        "title": "비즈니스 네트워킹",
        "emoji": "🌐",
        "description": "비즈니스 행사에서 새로운 사람 만나기",
        "user_role": "행사 참석자",
        "ai_role": "업계 전문가",
        "context": "You are a professional at a business networking event. Engage in casual professional small talk with the user.",
        "opening": "Hi there! I don't think we've met before. I'm Alex, I work in fintech. What brings you to this event today?",
    },
    {
        "id": "conference_call",
        "title": "전화/화상 회의",
        "emoji": "📞",
        "description": "해외 팀과의 원격 회의 진행",
        "user_role": "프로젝트 담당자",
        "ai_role": "해외 지사 매니저",
        "context": "You are a manager at an overseas branch on a video conference call. The user is presenting a project update. Occasionally mention connection issues or ask them to repeat/clarify, as happens in real remote meetings.",
        "opening": "Hi, can you hear me okay? Great. So, I wanted to check in on the project status. Could you give me a quick update on where things stand?",
    },
    {
        "id": "smalltalk",
        "title": "외국인 동료와 스몰토크",
        "emoji": "☕",
        "description": "탕비실/점심시간의 가벼운 직장 대화",
        "user_role": "직장 동료",
        "ai_role": "외국인 동료",
        "context": "You are a friendly foreign coworker chatting with the user in the office kitchen or at lunch. Keep it light — weekends, weather, office news, food. This practices casual-but-professional workplace small talk.",
        "opening": "Hey! Long week, huh? Got any plans for the weekend?",
    },
    {
        "id": "business_trip",
        "title": "해외 출장",
        "emoji": "✈️",
        "description": "출장지에서 바이어 미팅 및 식사 대접",
        "user_role": "출장 온 한국 회사 직원",
        "ai_role": "현지 바이어",
        "context": "You are a local buyer meeting the user who is on a business trip. Mix business discussion with dinner-table conversation — this practices the social side of business English.",
        "opening": "Welcome! I hope your flight wasn't too tiring. Shall we grab dinner and talk business? I know a great place nearby.",
    },
    {
        "id": "feedback_session",
        "title": "피드백 주고받기",
        "emoji": "💬",
        "description": "상사와의 1:1 면담 및 성과 리뷰",
        "user_role": "팀원",
        "ai_role": "직속 상사",
        "context": "You are the user's direct manager in a one-on-one performance review. Give both praise and constructive criticism, and ask the user about their goals and challenges. This practices receiving feedback and advocating for oneself professionally.",
        "opening": "Thanks for making time today. Overall you've been doing well this quarter. Before I share my feedback, how do you feel things have been going?",
    },
]


class AIHandler:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = "gpt-4o"

    def get_real_english_situations(self) -> list:
        return REAL_ENGLISH_SITUATIONS

    def get_business_scenarios(self) -> list:
        return BUSINESS_SCENARIOS

    def get_business_phrases(self) -> list:
        return BUSINESS_PHRASES

    def get_training_curriculum(self) -> list:
        return TRAINING_CURRICULUM

    def get_initial_greeting(self, past_history: list = None, mode: str = "free", scenario: dict = None, phrase_category: str = None, day_info: dict = None, sentences_done: int = 0, used_sentences: list = None, phrase_data: dict = None) -> dict:
        if mode == "phrase" and phrase_data:
            system = PHRASE_SYSTEM_PROMPT.format(
                phrase=phrase_data["phrase"],
                meaning=phrase_data["meaning"],
                usage=phrase_data["usage"],
                example1=phrase_data["examples"][0],
                example2=phrase_data["examples"][1],
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f'Start teaching "{phrase_data["phrase"]}". Present it warmly with full explanation, then ask user to repeat it.'},
            ]
        elif mode == "training":
            _day_info = day_info if day_info else TRAINING_CURRICULUM[0]
            _used = used_sentences or []
            used_str = "\n".join(f"- {s}" for s in _used) if _used else "(없음)"
            system = TRAINING_SYSTEM_PROMPT.format(
                day=_day_info["day"],
                focus=_day_info["focus"],
                pattern=_day_info["pattern"],
                sentences_done=sentences_done,
                used_sentences=used_str,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f'Start the training session. Evaluation: "start". Give a very brief 1-sentence welcome and immediately provide the first Korean sentence for Day {_day_info["day"]}: {_day_info["focus"]}.'},
            ]
        elif mode == "business" and scenario:
            system = BUSINESS_SYSTEM_PROMPT + f"\n\nYour role: {scenario['ai_role']}\nScenario context: {scenario['context']}"
            messages = [{"role": "system", "content": system}]
            if past_history:
                messages.extend(past_history[-10:])
            messages.append({"role": "user", "content": f"""Start the business scenario with your opening line.
Respond ONLY in JSON format:
{{
    "response": "{scenario['opening']}",
    "has_correction": false,
    "original": "",
    "corrected": "",
    "explanation": "",
    "pronunciation_tip": "",
    "hint": "첫 번째 대답 힌트 (한국어)"
}}"""})
        elif mode == "real_english" and scenario:
            system = REAL_ENGLISH_SYSTEM_PROMPT + f"\n\nToday's situation: {scenario['title']}\nContext: {scenario['context']}\nStart by briefly introducing what you'll practice in this situation, then dive into a natural conversation example and invite the student to respond."
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f'Start the {scenario["title"]} session. Give a warm, casual American-style intro and immediately model a natural expression from this situation, then invite the student to try. Respond in JSON format.'},
            ]
        elif mode == "elem_low":
            messages = [
                {"role": "system", "content": ELEM_LOW_SYSTEM_PROMPT},
                {"role": "user", "content": 'Start the class! Give a very warm, fun greeting for a young student (grades 1-3). Use simple English. Introduce yourself as their English teacher and ask ONE easy question to get them talking. Respond in JSON format.'},
            ]
        elif mode == "elem_high":
            messages = [
                {"role": "system", "content": ELEM_HIGH_SYSTEM_PROMPT},
                {"role": "user", "content": 'Start the class! Give a friendly greeting for an upper elementary student (grades 4-6). Introduce yourself as their English teacher and ask ONE interesting question to get them talking. Respond in JSON format.'},
            ]
        elif mode == "middle":
            messages = [
                {"role": "system", "content": MIDDLE_SYSTEM_PROMPT},
                {"role": "user", "content": 'Start the English class! Greet a middle school student (grades 7-9) in a cool, relatable way. Introduce yourself and ask ONE engaging question about their life, interests, or school. Respond in JSON format.'},
            ]
        elif mode == "high":
            messages = [
                {"role": "system", "content": HIGH_SYSTEM_PROMPT},
                {"role": "user", "content": 'Start the English class! Greet a high school student in a mature, engaging way. Introduce yourself as their conversation partner and pose ONE thought-provoking question to spark a real discussion. Respond in JSON format.'},
            ]
        elif past_history:
            prompt = """You are greeting a returning user who has practiced English with you before.
Greet them with 1 short sentence + 1 easy conversation-starter question — use their name if you know it, and reference something from past conversations if relevant.
Respond ONLY in this JSON format:
{
    "response": "One short, warm welcome-back sentence + one question to start the conversation",
    "has_correction": false,
    "original": "",
    "corrected": "",
    "explanation": "",
    "pronunciation_tip": "",
    "hint": "질문에 답할 때 쓸 수 있는 시작 표현 1개 (한국어 설명 포함, 예: \\"'Not much, just...' — 별일 없었다고 할 때\\")"
}"""
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(past_history[-20:])
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": """You are starting an English conversation session with a new user.
Respond ONLY in this JSON format:
{
    "response": "One short, friendly greeting sentence + one easy question to get them talking",
    "has_correction": false,
    "original": "",
    "corrected": "",
    "explanation": "",
    "pronunciation_tip": "",
    "hint": "질문에 답할 때 쓸 수 있는 시작 표현 1개 (한국어 설명 포함)"
}"""}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception:
            if mode == "phrase":
                fallback = "Hello! Let's practice some useful business expressions together. I'll teach you one phrase at a time!"
            elif mode == "business" and scenario:
                fallback = scenario["opening"]
            elif mode == "real_english":
                fallback = "Hey! Let's talk about how Americans really speak. What situation do you want to practice?"
            elif mode == "elem_low":
                fallback = "Hello! I'm your English teacher! 😊 What is your name?"
            elif mode == "elem_high":
                fallback = "Hi there! I'm your English teacher! What did you do today?"
            elif mode == "middle":
                fallback = "Hey! I'm your English teacher. What's been on your mind lately?"
            elif mode == "high":
                fallback = "Hello! I'm your English conversation partner. What topic would you like to explore today?"
            elif mode == "training":
                _day_info = day_info if day_info else TRAINING_CURRICULUM[0]
                fallback = f"Welcome to Day {_day_info['day']} training! Let's get started."
            else:
                fallback = "Hello! Ready to practice English?"
            return {
                "response": fallback,
                "has_correction": False,
                "original": "", "corrected": "", "explanation": "",
                "pronunciation_tip": "",
                "hint": "준비됐으면 \"I'm ready!\"라고 말해보세요." if mode == "phrase" else "",
                "current_phrase": "", "phrase_meaning": "",
                "evaluation": "start" if mode == "training" else "",
                "feedback_kr": "",
                "correct_answer": "",
                "next_prompt": {"korean": "", "hint": ""},
                "progress": {"current": 0, "total": 8},
                "session_complete": False,
            }

    def chat(self, user_message: str, conversation_history: list, mode: str = "free", scenario: dict = None, phrase_category: str = None, phrase_data: dict = None, day_info: dict = None, sentences_done: int = 0, used_sentences: list = None, opic_day: dict = None) -> dict:
        if mode == "opic" and opic_day:
            vocab_str = ", ".join(f"{v['word']} ({v['meaning']})" for v in opic_day["vocab"])
            expr_str = " / ".join(e["phrase"] for e in opic_day["expressions"])
            system = OPIC_SYSTEM_PROMPT.format(
                day=opic_day["day"],
                topic=opic_day["topic"],
                qtype=opic_day["type"],
                question=opic_day["question"],
                vocab=vocab_str,
                expressions=expr_str,
            )
        elif mode == "training":
            _day_info = day_info if day_info else TRAINING_CURRICULUM[0]
            _used = used_sentences or []
            used_str = "\n".join(f"- {s}" for s in _used) if _used else "(없음)"
            system = TRAINING_SYSTEM_PROMPT.format(
                day=_day_info["day"],
                focus=_day_info["focus"],
                pattern=_day_info["pattern"],
                sentences_done=sentences_done,
                used_sentences=used_str,
            )
        elif mode == "phrase":
            if phrase_data:
                system = PHRASE_SYSTEM_PROMPT.format(
                    phrase=phrase_data["phrase"],
                    meaning=phrase_data["meaning"],
                    usage=phrase_data["usage"],
                    example1=phrase_data["examples"][0],
                    example2=phrase_data["examples"][1],
                )
            else:
                system = PHRASE_SYSTEM_PROMPT.format(phrase="", meaning="", usage="", example1="", example2="")
        elif mode == "business" and scenario:
            system = BUSINESS_SYSTEM_PROMPT + f"\n\nYour role: {scenario['ai_role']}\nScenario context: {scenario['context']}"
        elif mode == "real_english":
            system = REAL_ENGLISH_SYSTEM_PROMPT
            if scenario:
                system += f"\n\nSituation focus: {scenario['title']} — {scenario['context']}"
        elif mode == "elem_low":
            system = ELEM_LOW_SYSTEM_PROMPT
        elif mode == "elem_high":
            system = ELEM_HIGH_SYSTEM_PROMPT
        elif mode == "middle":
            system = MIDDLE_SYSTEM_PROMPT
        elif mode == "high":
            system = HIGH_SYSTEM_PROMPT
        else:
            system = SYSTEM_PROMPT

        messages = [{"role": "system", "content": system}]
        # phrase 모드는 이전 대화 오염 방지를 위해 history 사용 안 함
        history_limit = 0 if mode == "phrase" else 20
        messages.extend(conversation_history[-history_limit:])
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            response_format={"type": "json_object"},
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, content: str) -> dict:
        try:
            data = json.loads(content)
            return {
                "response": data.get("response", "I'm sorry, could you repeat that?"),
                "has_correction": bool(data.get("has_correction", False)),
                "original": data.get("original", ""),
                "corrected": data.get("corrected", ""),
                "explanation": data.get("explanation", ""),
                "pronunciation_tip": data.get("pronunciation_tip", ""),
                "hint": data.get("hint", ""),
                "current_phrase": data.get("current_phrase", ""),
                "phrase_meaning": data.get("phrase_meaning", ""),
                "evaluation": data.get("evaluation", ""),
                "feedback_kr": data.get("feedback_kr", ""),
                "correct_answer": data.get("correct_answer", ""),
                "next_prompt": data.get("next_prompt", {"korean": "", "hint": ""}),
                "progress": data.get("progress", {"current": 0, "total": 8}),
                "session_complete": data.get("session_complete", False),
                "phrase_confirmed": bool(data.get("phrase_confirmed", False)),
                "opic_feedback": data.get("opic_feedback", ""),
                "better_expression": data.get("better_expression", ""),
                "better_explanation": data.get("better_explanation", ""),
            }
        except (json.JSONDecodeError, KeyError):
            text = re.sub(r'\{.*\}', '', content, flags=re.DOTALL).strip()
            return {
                "response": text or content,
                "has_correction": False,
                "original": "", "corrected": "", "explanation": "",
                "pronunciation_tip": "", "hint": "",
            }


TRAINING_CURRICULUM = [
    # Week 1: 기본 현재형
    {"day": 1, "focus": "I am + 형용사/명사", "pattern": "I am [adjective/noun]", "level": 1},
    {"day": 2, "focus": "I have + 명사", "pattern": "I have [noun]", "level": 1},
    {"day": 3, "focus": "I like / I love / I hate", "pattern": "I like/love/hate [noun/verb-ing]", "level": 1},
    {"day": 4, "focus": "I want + 명사/to 동사", "pattern": "I want [noun] / I want to [verb]", "level": 1},
    {"day": 5, "focus": "I need + 명사/to 동사", "pattern": "I need [noun] / I need to [verb]", "level": 1},
    {"day": 6, "focus": "I go / I work / I live", "pattern": "I [simple verb] [place/time]", "level": 1},
    {"day": 7, "focus": "Week 1 복습", "pattern": "Mix of Week 1 patterns", "level": 1},
    # Week 2: 의문문 & 부정문
    {"day": 8, "focus": "Do you...? / Does she...?", "pattern": "Do/Does [subject] [verb]?", "level": 2},
    {"day": 9, "focus": "I don't / I doesn't", "pattern": "I don't [verb]", "level": 2},
    {"day": 10, "focus": "What / Where / When is...?", "pattern": "What/Where/When is [noun]?", "level": 2},
    {"day": 11, "focus": "Can you...? / I can...", "pattern": "Can [subject] [verb]?", "level": 2},
    {"day": 12, "focus": "There is / There are", "pattern": "There is/are [noun]", "level": 2},
    {"day": 13, "focus": "It is + 형용사 + to 동사", "pattern": "It is [adj] to [verb]", "level": 2},
    {"day": 14, "focus": "Week 2 복습", "pattern": "Mix of Week 2 patterns", "level": 2},
    # Week 3: 과거형
    {"day": 15, "focus": "I was / I were", "pattern": "I was [adjective/noun]", "level": 3},
    {"day": 16, "focus": "I went / I had / I did", "pattern": "I [past verb] [object]", "level": 3},
    {"day": 17, "focus": "Did you...? / I didn't...", "pattern": "Did [subject] [verb]? / I didn't [verb]", "level": 3},
    {"day": 18, "focus": "I was -ing (과거 진행)", "pattern": "I was [verb]-ing", "level": 3},
    {"day": 19, "focus": "When I..., I... (시간 접속사)", "pattern": "When I [past], I [past]", "level": 3},
    {"day": 20, "focus": "I've been / I've done (현재완료)", "pattern": "I have [past participle]", "level": 3},
    {"day": 21, "focus": "Week 3 복습", "pattern": "Mix of Week 3 patterns", "level": 3},
    # Week 4: 의견 & 감정 표현
    {"day": 22, "focus": "I think / I believe", "pattern": "I think/believe that [clause]", "level": 4},
    {"day": 23, "focus": "I would like to / I'd like", "pattern": "I would like to [verb]", "level": 4},
    {"day": 24, "focus": "Could you...? / Would you mind...?", "pattern": "Could you / Would you mind [verb]?", "level": 4},
    {"day": 25, "focus": "I'm planning to / I'm going to", "pattern": "I'm planning/going to [verb]", "level": 4},
    {"day": 26, "focus": "I'm not sure if / I wonder if", "pattern": "I'm not sure if / I wonder if [clause]", "level": 4},
    {"day": 27, "focus": "If + 조건절", "pattern": "If I [verb], I will [verb]", "level": 4},
    {"day": 28, "focus": "비즈니스: Regarding / As for", "pattern": "Regarding/As for [topic], [statement]", "level": 4},
    {"day": 29, "focus": "비즈니스: I'd like to propose / I suggest", "pattern": "I'd like to propose / I suggest [noun/verb-ing]", "level": 4},
    {"day": 30, "focus": "최종 복습 & 종합", "pattern": "All patterns combined", "level": 5},
]


TRAINING_SYSTEM_PROMPT = """You are a Korean-to-English sentence training coach for a Korean beginner who can read English but cannot produce sentences yet.
Your mission: build their sentence production ability through daily drilling.

ALWAYS respond in this EXACT JSON format, no exceptions:
{{
    "response": "brief encouraging feedback in English (1 sentence)",
    "evaluation": "correct" OR "partial" OR "incorrect" OR "start",
    "feedback_kr": "평가 및 설명 한국어 2-3문장. 왜 맞는지/틀린지 명확히 설명.",
    "correct_answer": "The correct English translation of the Korean sentence",
    "next_prompt": {{
        "korean": "다음에 번역할 한국어 문장 (마지막 문장이면 빈 문자열)",
        "hint": "패턴 힌트 (예: I am + 형용사)"
    }},
    "progress": {{"current": 0, "total": 8}},
    "session_complete": false,
    "has_correction": true OR false,
    "original": "user's input exactly",
    "corrected": "corrected English (empty string if correct)",
    "explanation": "correction explanation in Korean (empty string if correct)"
}}

Today's focus: Day {day} - {focus}
Pattern: {pattern}
Sentences completed this session: {sentences_done} / 8

ALREADY MASTERED - DO NOT USE THESE SENTENCES AGAIN:
{used_sentences}

RULES:
1. Generate Korean sentences that naturally require today's pattern to translate
2. NEVER repeat any sentence from the "ALREADY MASTERED" list above - create completely different sentences
3. Start simple, gradually increase complexity across 8 sentences
4. Keep sentences practical: work, daily life, feelings, opinions
5. evaluation must be "correct" if the meaning is right (minor errors ok), "partial" if structure is right but has errors, "incorrect" if pattern is wrong
6. feedback_kr: always explain WHY, reference the pattern
7. When sentences_done >= 7 (user just answered the 8th), set session_complete=true and next_prompt.korean=""
8. On "start" (first message), give the very first Korean sentence with a warm brief intro
9. Be very encouraging - this is a beginner!"""
