import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=MODEL,
    temperature=0,
    streaming=True,
)

embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=api_key,
    openai_api_base=base_url,
    check_embedding_ctx_length=False,
)
