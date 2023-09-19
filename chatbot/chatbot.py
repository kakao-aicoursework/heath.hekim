"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.agents.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

###########################################################
# Instances
llm = OpenAI(temperature=0)

def get_kakao_sync_doc():

    with open("project_data_kakao_sync.txt", "r", encoding='UTF-8') as f:
        return f.read()

tools =[
        Tool(
            name="doc",
            func=get_kakao_sync_doc,
            description="카카오 싱크에 대한 설명을 가져옵니다.",
        ),
]

#agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#chat = ChatOpenAI(temperature=0.8)
#system_message = "user의 질문을 다음 문서를 참고하여 대답하세요:\n{}".format(get_kakao_sync_doc())
#system_message_prompt = SystemMessage(content=system_message)
#human_template = ("question: {query}")
#human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#chain = LLMChain(llm=chat, prompt=chat_prompt)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def truncate_text(text, max_tokens=3000):
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:  # 토큰 수가 이미 3000 이하라면 전체 텍스트 반환
        return text
    return enc.decode(tokens[:max_tokens])

llm = OpenAI(temperature=0.9)
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    user의 주어진 질문을 다음 문서를 참고하여 대답하세요:
    질문: {query}
    문서: {context}
    """
)

chain = LLMChain(llm=llm, prompt=prompt)


class State(pc.State):
    """The app state."""

    prompt: str
    chat_history :list[str] = []

    def do_ask(self):
        self.chat_history.append(self.prompt)
        #self.chat_history.append(agent.run(self.prompt))
        short_context = truncate_text(get_kakao_sync_doc(), max_tokens=3500) 
        print("prompt={}, context={}".format(self.prompt, short_context))
        answer = chain.run(query=self.prompt, context=short_context)
        self.chat_history.append(answer)

    def set_prompt(self, prompt):
        self.prompt = prompt

    @pc.var
    def get_history(self):
        return '\n'.join(self.chat_history)


def index():
    return pc.center(
        pc.vstack(
            pc.center(
                pc.vstack(
                    pc.heading("chatbot", font_size="1.5em"),
                    pc.text_area(
                        placeholder="GPT Result",
                        height="30em",
                        width="100%",
                        is_read_only=True,
                        text = State.get_history,
                    ),
                    pc.input(
                        placeholder="Question",
                        on_blur=State.set_prompt,
                    ),
                    pc.button("Ask", on_click=State.do_ask),
                    shadow="lg",
                    padding="1em",
                    border_radius="lg",
                    width="100%",
                ),
                width="100%",
            ),
            width="50%",
            spacing="2em",
        ),
        padding_top="6em",
        text_align="top",
        position="relative",
    )
 

# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="llm chatbot")
app.compile()
