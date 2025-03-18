from fastapi import FastAPI
from agno.agent import Agent, AgentKnowledge
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase

from agno.vectordb.pgvector import PgVector
from agno.storage.agent.postgres import PostgresAgentStorage
import os
from agno.embedder.voyageai import VoyageAIEmbedder
from agno.storage.agent.json import JsonAgentStorage
from agno.models.groq import Groq


app = FastAPI()

from agno.vectordb.pineconedb import PineconeDb

api_key = os.getenv("PINECONE_API_KEY")
index_name = "unified-pyano"

vector_db = PineconeDb(
    name=index_name,
    dimension=1024,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=api_key,
    use_hybrid_search=True,
    hybrid_alpha=0.5,
    embedder=VoyageAIEmbedder(),
    # namespace="test_org"
)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://divbk.s3.ap-south-1.amazonaws.com/Divyraj+Resume.pdf"],
    vector_db=vector_db,
)

# knowledge_base = AgentKnowledge(
#     vector_db=vector_db,
#     )

knowledge_base.load(recreate=False, upsert=True)

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a expert on docs",
    knowledge=knowledge_base,
    storage=JsonAgentStorage(dir_path="tmp/agent_sessions_json"),
    markdown=True,
)

@app.get("/ask")
async def ask(query: str):
    response = agent.run(query)
    return {"response": response.content}