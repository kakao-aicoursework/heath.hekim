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

from langchain.document_loaders import (
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = "chroma-persist"
CHROMA_COLLECTION_NAME = "chroma_database"

###########################################################
# Instances
llm = ChatOpenAI(temperature=0.1, max_tokens=2000)

db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
retriever = db.as_retriever()

def prepare_embedding_db(file_path):
    documents = TextLoader(file_path, encoding='UTF-8').load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

intent_chain = LLMChain(llm=llm,  prompt=ChatPromptTemplate.from_template(template="""
Your job is to determine the user message is related to the following list:
- 카카오싱크
- 카카오 싱크
- KakaoSync
- Kakao Sync

Considering the user message, then return answer "YES" if it's related to the list, "NO" if it's not.


User: {user_message}
Answer:
"""),  output_key="proceed", verbose=True,)

search_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template(template="""
Your job is to read the following related document and answer the user's question based on the document.
                                                                          
<related_documents>
{related_documents}
</related_documents>


User's question: {user_message}
Answer:
"""),  output_key="answer", verbose=True,)

default_chain = ConversationChain(llm=llm, output_key="answer")

def get_kakao_sync_doc():

    with open("project_data_kakao_sync.txt", "r", encoding='UTF-8') as f:
        return f.read()

prepare_embedding_db("project_data_kakao_sync.txt")

from pprint import pprint

def ask_prompt(prompt):
    context = dict(user_message=prompt)
    context["input"] = context["user_message"]

    proceed = intent_chain.run(context)

    if proceed  == "YES":
        context["related_documents"] = query_db(context["user_message"], use_retriever=True)

        answer = search_chain(context)
    else:
        answer = {"answer": default_chain.run(context["user_message"])}

    return answer

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = retriever.get_relevant_documents(query)
    else:
        docs = db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]

    return str_docs


class State(pc.State):
    """The app state."""

    prompt: str
    chat_history : str

    def do_ask(self):
        self.chat_history = self.chat_history + "\n" + self.prompt
        #self.chat_history.append(agent.run(self.prompt))
        result = ask_prompt(self.prompt)
        print(result["answer"])
        self.chat_history = self.chat_history + "\n" + result["answer"]
        self.prompt = ""

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
                        value = State.chat_history
                    ),
                    pc.input(
                        placeholder="Question",
                        on_change=State.set_prompt,
                        value=State.prompt,
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
