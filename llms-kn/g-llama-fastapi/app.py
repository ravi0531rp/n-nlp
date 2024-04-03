from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)


llm=Ollama(model="phi")

prompt=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 10 sentences. Also, format it line by line. Use alliteration.")


add_routes(
    app,
    prompt|llm,
    path="/poem"


)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
