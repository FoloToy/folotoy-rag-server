import os
import dotenv
import time
import uvicorn

from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import CharacterTextSplitter
from sse_starlette import EventSourceResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

dotenv.load_dotenv()

DEFAULTS = {
    'CHAIN_TYPE': "map_reduce"
}


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def summarize(content: str, chain_type: str):
    llm = OpenAI(temperature=0)
    text_splitter = CharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0,
        length_function=len,
    )
    summary_chain = load_summarize_chain(llm, chain_type=chain_type)
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain, text_splitter=text_splitter)
    summary_text = summarize_document_chain.invoke(content)
    return summary_text


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


def predict(query: str, model_id: str, chain_type: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    summary = summarize(query, chain_type)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=summary['output_text'], role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))
    yield '[DONE]'


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    user_content = request.messages[-1].content
    chain_type = get_env("CHAIN_TYPE")
    generate = predict(user_content, request.model, chain_type)
    return EventSourceResponse(generate, media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)