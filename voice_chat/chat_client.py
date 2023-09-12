import websockets
import asyncio
import base64
import json
from configure import auth_key
import streamlit as st
import requests
import re
import argparse
from typing import List, Union

from llama.tokenizer import Tokenizer as tok

if 'run' not in st.session_state:
	st.session_state['run'] = False
import pyaudio

FRAMES_PER_BUFFER = 3200
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
p=pyaudio.PyAudio()

# start recorder

stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
        )

if 'run' not in st.session_state:
    st.session_state['run']=False

def start_listening():
	st.session_state['run']=True

def stop_listening():
	st.session_state['run']=False

st.title('My first voice app')
start, stop = st.columns(2)

start.button('Start', on_click=start_listening)
stop.button('Stop', on_click=stop_listening)

URL = f'wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}'



def token_counter(model_path: str):
	'''Instantiate the tokenizer so we can keep track of the conversion history length in tokens.
	args:
		model_path: str: path to the sentancepiece model use for tokenizing
	returns:
		function: partial function that tokenizes the input text and returns the length in tokens.
	'''
	tokenizer = tok(model_path)
	def token_count(prompt: Union[List,str]):
		if isinstance(prompt,str):
			prompt=[prompt]
		str_prompt = ''.join([json.dumps(a) for a in prompt])
		return len(tokenizer.encode(str_prompt, False,False ))
	return token_count

# use async functions to send input from mic and one for listening for response. 
async def send_receive(args):

	print(f'Connecting websocket to url ${URL}')


	async with websockets.connect(
		URL,
		extra_headers=(("Authorization", auth_key),),
		ping_interval=5,
		ping_timeout=20
	) as _ws:

		r= await asyncio.sleep(0.1)
		print("Receiving SessionBegins ...")

		session_begins = await _ws.recv()
		print(session_begins)
		print("Sending messages ...")


		async def send(args):
			while st.session_state['run']:
				try:
					data = stream.read(FRAMES_PER_BUFFER)
					data = base64.b64encode(data).decode("utf-8")
					json_data = json.dumps({"audio_data":str(data)})
					r= await _ws.send(json_data)

				except websockets.exceptions.ConnectionClosedError as e:
					print(e)
					assert e.code == 4008
					break

				except Exception as e:
					assert False, "Not a websocket 4008 error"

				r= await asyncio.sleep(0.01)
		  
			return True
	  

		async def receive(args):
			
			system_prompt=None # Sytem prompt to be used to instruct Lllama2 how to respond
			chat_history = [] # Accumulated conversation.
			chat_history_length = [] # the number of tokens in teh accumulated chat history for each turn.

			tok_counter = token_counter(args.tokenizer_model_path) # functino that counts tokens in the chat_history

			sys_keywords = args.system_keywords.lower().strip()
			
			while st.session_state['run']:
				try:
					result_str = await _ws.recv()
					if json.loads(result_str)['message_type']=='FinalTranscript' and json.loads(result_str)['text']!="" :
						st.markdown(json.loads(result_str)['text']+'*')
						# If 'system prompt' keyword, then store it and use it when submitting a dialog to the chat bot.
						if sys_keywords in json.loads(result_str)['text'].lower():
							system_prompt={"role":"system","content":f"{re.sub(sys_keywords,'',json.loads(result_str)['text']).strip()}"}
						else:
							st.markdown('sending to chatbot...')
							if system_prompt is not None:
								# prepend the system prompt to the dialog.
								# Note that the llama2 model appears to keep track of the prior conversation ( up to the context window length - TBC)
								prompt = [system_prompt,{"role":"user","content":f"{json.loads(result_str)['text']}"}]
							else:
								prompt = [{"role":"user","content":f"{json.loads(result_str)['text']}"}]
							
							# track the conversation history
							if chat_history:
								chat_history.extend(prompt)
								chat_history_length.append(tok_counter(prompt))
							else:
								chat_history_length=[tok_counter(prompt)]
								chat_history = prompt

							# maintain chat history 'queue' length
							while sum(chat_history_length)>args.chat_history_length:
								chat_history=chat_history[1:]
								chat_history_length=chat_history_length[1:]
														
							r = requests.post(url=args.llm_endpoint, json={"dialogs":chat_history})
							data = r.json()
							chat_response=data['data'][0]['generation']['content']
							st.markdown(chat_response)

							# add response to the history
							chat_history.extend([{"role":"assistant","content":chat_response}])

							#reset system prompt as we don't need to include it for every turn of the conversation
							system_prompt = None
						
				except websockets.exceptions.ConnectionClosedError as e:
					print(e)
					assert e.code == 4008
					break

				except Exception as e:
					assert False, "Not a websocket 4008 error"
	  
		send_result, receive_result = await asyncio.gather(send(args), receive(args))






if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_endpoint', type=str, default = 'http://localhost:8080/chat', help='URL for REST API serving LLM')
	parser.add_argument('--system_keywords',type=str,default='system prompt', help='phrase used to start setting of system prompt')
	parser.add_argument('--chat_history_length',type=int, default=3000, help='The number tokens in the context window that available to store conversation history')
	parser.add_argument('--tokenizer_model_path', type=str, default='./tokenizer.model',help='used to calculate the tokens in the conversation history')

	args = parser.parse_args()

	asyncio.run(send_receive(args))