import os
import dotenv
import time
import uvicorn

from openai import OpenAI
from sse_starlette import EventSourceResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from langdetect import detect

dotenv.load_dotenv()

DEFAULTS = {
    'HTTPX_TIMEOUT': 60,
    'TEMPERATURE': 0,
    'MAX_TOKENS': 4096
}

summary_refine_prompt_template = """\
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary.
If the context isn't useful, return the original summary.
The language of summary must keep in {language}.
"""

summary_prompt_template = """Write a concise summary of the following, 
and the language of summary must keep in {language}.


"{text}"


CONCISE SUMMARY:"""

todo_refine_prompt_template = """\
Your job is to produce a final todo list.
We have provided an existing todo list up to a certain point: {answer}
We have the opportunity to refine the existing todo list (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original todo list.
If the context isn't useful, return the original todo list.
The language of todo list must keep in {language}.
"""

todo_prompt_template = """Write a concise todo list of the following, 
and the language of todo list must keep in {language}:


"{text}"


CONCISE TODO LIST:"""


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def make_todo_list(content: str, model_name: str):
    client = OpenAI()

    language = detect(content)
    length = len(content)
    chunk_size = 1500
    start_idx = 0
    end_idx = 0
    times = 1
    answer = None
    while end_idx < length:
        end_idx = start_idx + chunk_size
        if end_idx >= length:
            end_idx = length

        text = content[start_idx:end_idx]
        text_nolines = text.replace("\n", "\\n")
        print(f"idx=[{start_idx}, {end_idx}], text: {text_nolines}")
        start_idx = end_idx

        if times == 1:
            content = todo_prompt_template.format(text=text, language=language)
        else:
            content = todo_refine_prompt_template.format(answer=answer, text=text, language=language)

        messages = [{
            "role": "user",
            "content": content
        }]
        params = dict(
            messages=messages,
            stream=False,
            model=model_name,
            temperature=get_env("TEMPERATURE"),
            max_tokens=get_env("MAX_TOKENS"),
            timeout=get_env("HTTPX_TIMEOUT")
        )

        chat_completion = client.chat.completions.create(**params)
        answer = chat_completion.choices[0].message.content
        print(f"Todo times: {times}, answer: {answer}")
        times = times + 1

    return answer


def summarize(content: str, model_name: str):
    client = OpenAI()

    language = detect(content)
    length = len(content)
    chunk_size = 1500
    start_idx = 0
    end_idx = 0
    times = 1
    answer = None
    while end_idx < length:
        end_idx = start_idx + chunk_size
        if end_idx >= length:
            end_idx = length

        text = content[start_idx:end_idx]
        text_nolines = text.replace("\n", "\\n")
        print(f"idx=[{start_idx}, {end_idx}], text: {text_nolines}")
        start_idx = end_idx

        if times == 1:
            content = summary_prompt_template.format(text=text, language=language)
        else:
            content = summary_refine_prompt_template.format(answer=answer, text=text, language=language)

        messages = [{
            "role": "user",
            "content": content
        }]
        params = dict(
            messages=messages,
            stream=False,
            model=model_name,
            temperature=get_env("TEMPERATURE"),
            max_tokens=get_env("MAX_TOKENS"),
            timeout=get_env("HTTPX_TIMEOUT")
        )

        chat_completion = client.chat.completions.create(**params)
        answer = chat_completion.choices[0].message.content
        print(f"Summarize times: {times}, answer: {answer}")
        times = times + 1

    return answer


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


def predict(query: str, model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    summary = summarize(query, model_id)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=f"Summary:\n {summary}", role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
        choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True))

    todo_list = make_todo_list(query, model_id)
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=f"Todo List:\n {todo_list}", role="assistant"),
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
    generate = predict(user_content, request.model)
    return EventSourceResponse(generate, media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)