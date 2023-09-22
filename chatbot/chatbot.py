"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory

###########################################################
# constants
CHROMA_PERSIST_DIR = "chroma-persist"
CHROMA_COLLECTION_NAME = "chroma_database"

###########################################################
# Instances
llm = ChatOpenAI(temperature=0.1, max_tokens=2000)

db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=OpenAIEmbeddings(), collection_name=CHROMA_COLLECTION_NAME,)

retriever = db.as_retriever()

###########################################################
# prepare embedding db
def prepare_embedding_file(file_path):
    documents = TextLoader(file_path, encoding='UTF-8').load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

def prepare_embedding_dir(dir_path):

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                try:
                    prepare_embedding_file(file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = retriever.get_relevant_documents(query)
    else:
        docs = db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]

    return str_docs

###########################################################
# LLM chains

# 1. intent chain
intent_chain = LLMChain(llm=llm,  prompt=ChatPromptTemplate.from_template(template="""
Your job is to determine the user message is related to the following list:
- 카카오 소셜
- 카카오싱크
- 카카오톡 채널

Considering the user message, if the message is related to '카카오 소셜', then answer as 'kakao_social',
if the message is related to '카카오싱크', then answer as 'kakao_sync',
if the message is related to '카카오톡 채널', then answer as 'kakao_channel',
if the message is related to none of them, then answer as 'others'.

User: {user_message}
Answer:
"""),  output_key="intent", verbose=True,)

# 2. default query chain
search_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template(template="""
Prev conversation:
{history}

Your job is to read the following related document and answer the user's question based on the document.
                                                                          
<related_documents>
{related_documents}
</related_documents>

User's question: {user_message}
Answer:
"""),  output_key="answer", verbose=True,)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)

default_chain = ConversationChain(llm=llm, memory=memory, output_key="answer")

# files into embedding db
prepare_embedding_dir(os.path.join(os.path.curdir, "files"))

def ask_prompt(prompt):
    context = dict(user_message=prompt)
    context["input"] = context["user_message"]

    intent = intent_chain.run(context)

    if intent == "kakao_social" or intent == "kakao_sync" or intent == "kakao_channel":
        context["related_documents"] = query_db(context["user_message"], use_retriever=True)
        context["history"] = memory.buffer

        answer = search_chain(context)
    else:
        answer = {"answer": default_chain.run(context["user_message"])}

    return answer

###########################################################
# the app implementation

class State(pc.State):
    """The app state."""

    prompt: str
    chat_history : str

    def do_ask(self):
        self.chat_history = self.chat_history + "\n\n" + self.prompt
        result = ask_prompt(self.prompt)
        print(result["answer"])
        self.chat_history = self.chat_history + "\n\n" + result["answer"]
        self.prompt = ""    # clear the prompt input edit

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
