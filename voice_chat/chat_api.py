from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
from llama import Llama

import argparse
import os
import json

app=FastAPI()

class ChatInferencePrompt(BaseModel):
    role: str
    content: str
    def __getitem__(self, item):
        return getattr(self,item)

class InferenceDialog(BaseModel):
    dialogs: List[List[ChatInferencePrompt]]
    def __getitem__(self, item):
        return getattr(self,item)

@app.post('/chat')
def perform_inference(dialogs: InferenceDialog):
    '''
    Expects a valid json string
    '''
    dialogs=dialogs['dialogs']
    print(dialogs)

    results = llm.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    return {"data":results}

def build_llm(args):
    llm = Llama.build(
        ckpt_dir= os.path.join(args.root_path,args.model_path),
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    return llm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, default='/home/mtman/Documents/Repos/llama/')
    parser.add_argument('--model_path', type=str, default='llama-2-7b', help='Relative path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer.model')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.4)
    parser.add_argument('--max_seq_len',type=int, default=2000)
    parser.add_argument('--max_gen_len',type=int, default=200)
    parser.add_argument('--max_batch_size',type=int, default=4)
    args = parser.parse_args()


    llm=build_llm(args)

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)