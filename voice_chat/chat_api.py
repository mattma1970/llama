from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from llama import Llama

import argparse
import os
import json
import logging
from dataclasses import dataclass

from griptape.structures import Agent
from griptape.utils import Chat #   <-- Added Chat
from griptape.drivers import LocalLlamaPromptDriver
from griptape.utils.prompt_stack import PromptStack

app=FastAPI()
logger = logging.getLogger(__name__)


class ChatInferencePrompt(BaseModel):
    role: str
    content: str
    def __getitem__(self, item):
        return getattr(self,item)

class InferenceDialog(BaseModel):
    # Important. Each inner list is a complete conversation, ending with a 'user' prompt. The LLM accepts multiple conversations.
    # This shouldnt be confused with the List[List[dict]] used to represent a single conversation in the chat app's chat_history.
    dialogs: List[List[ChatInferencePrompt]]
    def __getitem__(self, item):
        return getattr(self,item)

@dataclass
class SessionDialog:
    prompt: List[ChatInferencePrompt]
    session_id: str # persisted session_id


_ALL_TASKS=['chat_with_agent:post','chat:post','llm_params:get']

@app.post('/chat')
def perform_inference(dialogs: InferenceDialog):
    '''
        Post processes by default - DEPRECATED
    '''

    dialogs=dialogs['dialogs']

    response = llm.chat_completion(
        dialogs,  # List[List[dict]]
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.debug:
        for dialog, result in zip(dialogs, response):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

    response=response['data'][0]['generation']['content']

    return {"data":response}



@app.post('/chat_with_agent')
def chat_using_agent(dialogs: SessionDialog) -> Dict[str,str]:
    '''
        Chat text_generation using griptape agent.
        Conversation memory is managed by Griptape so only the new question is passed in. 
        Args:
            dialogs: SessionDialog[prompt:str, session_id:str)
                     prompt: List[ChatInferencePrompt]
                     session: str: uuid4, a persisted session_id
        Returns: {'data': results}: chat completion results as dictionary. Mimics response from direct call to the models API.      
    '''

    '''Retrieve the agent for the requestor. The agent also contains the conversation memory.'''
    session_id:str = dialogs['session_id']
    if session_id not in agent_registry:
        agent_registry[session_id] = build_agent(llm,args)
    agent: Agent = agent_registry[session_id] 


    if len(dialogs['prompt'])>1 and dialogs['prompt'][0].role==PromptStack.SYSTEM_ROLE:
        system_prompt = dialogs['prompt'][0]
        user_input=dialogs['prompt'][1].content
        raise Warning('Replacing System Prompt is not yet implemented.')
    else:
        system_prompt = None
        user_input=dialogs['prompt'][0].content
   
    #print(user_input)
    response = agent.run(user_input).output.to_text()
    return {'data':response}

@app.get('/llm_params')
def get_llm_params():
    '''Get some model params that are needed by the app for chat history management.'''
    return {
            'max_seq_len':args.max_seq_len,
            'max_gen_len': args.max_gen_len,
            'top_p':args.top_p,
            'max_batch_size':args.max_batch_size,
            'supported_tasks': _ALL_TASKS
            }

''' ===== AI-staff manufacturing facility ==== '''

def build_llm(args):
    llm = Llama.build(
        ckpt_dir= os.path.join(args.root_path,args.model_path),
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    return llm

def build_agent(model, args):   
    params =  {
                "max_new_tokens": args.max_gen_len, #new tokens per generation
                "max_tokens": args.max_seq_len, # maximum context window+new_tokens.
            }
    agent = Agent(logger_level=logging.ERROR, prompt_driver=LocalLlamaPromptDriver(inference_resource=model, task='chat', tokenizer_path=args.tokenizer_path, params = params))
    return agent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, default='/home/mtman/Documents/Repos/llama/')
    parser.add_argument('--model_path', type=str, default='llama-2-7b', help='Relative path to root_path')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer.model')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.4)
    parser.add_argument('--max_seq_len',type=int, default=2000)
    parser.add_argument('--max_gen_len',type=int, default=256)
    parser.add_argument('--max_batch_size',type=int, default=4)
    parser.add_argument('--debug',action='store_true', default =False, help='Be far more chatty about the internals.')
    args = parser.parse_args()

    llm=build_llm(args)
    agent = build_agent(llm,args)
    agent_registry = {}

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)