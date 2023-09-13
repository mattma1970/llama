'''
mattma1970@gmail.com 8/9/2023

This experiment combines the non-blocking I/O of webassembly.ai API with voice input sources via a broweser over webRTC. 
The streamlit package used is https://github.com/whitphx/streamlit-webrtc
The LLM used is a Llama 2 servers via a fastAPI endpoint on the same server as the chat app. 

'''
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av # Python bindings for ffmpeg
import cv2
import pydub  # for processing dataframes returned
import numpy as np

import websockets
import asyncio
import aiohttp
import base64
import json
import math 
import queue
from itertools import chain

import requests

from configure import auth_key

import requests
import re
import argparse
from typing import List, Union, Any, Callable
from uuid import uuid4
import time
from tqdm import tqdm

from registry import FuncRegistry
from voice_chat.audio_connections import WebRTCAudioSteam, PyAudioStream, AudioConnection
from utils import st_html


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


func_registry = FuncRegistry() # Registry for callbacks

from llama.tokenizer import Tokenizer as tok

#Globals
FRAMES_PER_BUFFER = 4800  # units = samples
MAX_FRAMES = 99 # units = Frames NOT samples. For streamlit_webrtc 1 frame=0.02 seconds. AssemblyAI, maximum duration of audio posted is 2 seconds.


#Session_state_keys
WEBRTC_CONNX_ESTABLISHED = 'webRTC_runtime_configuration_is_set' # Flag used  to indicate that the audio_settings of the inbound audio have been collected. This is done once when the connection is established.
WEBRTC_CONNECTION = 'connx' # persitance of the connection and audio data from the streamer.
CHAT_HISTORY = 'chat_dialogs' # list of list of dialogs to submit to LLM i.e. dialog history.
CHAT_HISTORY_LENGTHS = 'chat_history'
CONVERSATION='conversation' # clean text for display

if 'conversation' not in st.session_state:
	st.session_state['conversation']=' '

if CHAT_HISTORY not in st.session_state:
	st.session_state[CHAT_HISTORY]=[]
	st.session_state[CHAT_HISTORY_LENGTHS]=[]

#### Strealit UI ####
st.title('Sqwak: Voice chat with Llama2-7B')

# <head>
# CSS styling for streamlit elements.
styl = """
<style>
    .stButton{
        bottom: 2rem;
        left:500; 
        right:500;
        z-index:4;
    }
	[data-testid="stMarkdownContainer"]:has(div.statusText) div {
        bottom: 2rem;
        background-color: rgb(173, 216,230);
		width: 100%;
        border-top-right-radius: 6px; 
		border-bottom-right-radius: 6px;
        z-index:4;
		padding: 4px 4px 4px 10px;
	}
	[data-testid="stMarkdownContainer"]:has(div.statusLabel) div {
        bottom: 2rem;
        background-color: rgb(240,230,140);
		width: 100%;
        border-top-left-radius: 6px; 
		border-bottom-left-radius: 6px;
        z-index:4;
		text-align:centre;
		padding: 4px 4px 4px 10px;
	}

    @media screen and (max-width: 1000px) {
        .stTextInput {
            left:2%; 
            width: 100%;
            bottom: 2.1rem;  
            z-index:2; 
        }                
		.stMarkdownContainer {
            left:2%; 
            width: 100%;
            bottom: 2.1rem;  
            z-index:2; 
		}        
        .stButton {            
            left:2%;  
            width: 100%;       
            bottom:0rem;
            z-index:3; 
        }          
    } 

</style>

"""
st.markdown(styl, unsafe_allow_html=True)

js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
        for (let index = 0; index < textAreas.length; index++) {{
            textAreas[index].scrollTop = textAreas[index].scrollHeight;
        }}
    }}
    scroll({len(st.session_state['conversation'])})
</script>
"""
#</head>
#<body>

status_label, status_area = st.columns([0.2,0.8])
with status_label:
	label=st_html(st.empty(),'statusLabel')
	label.write('Status')

with status_area:
	st_status_bar = st_html(st.markdown('Startup..'),'statusText','Startup..')
	st_text_output = st.empty()
#</body>

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

def stt_b64_encode(a: av.AudioFrame, channels: int=2):
	'''
	Returns base64 encoded bytes from a single channel of the audioframe.
	@args:
		a: av.AudioFrame : A single AudioFrame that exposes a to_ndarray function for retrieving the raw bytes data.
	@returns:
		utf-8 encoded base64 data from dataframe.
	'''
	a=np.ascontiguousarray(a.to_ndarray()[0][0::channels]) # audio channels are interleaved.
	return base64.b64encode(a).decode('utf-8')


def turn_sum(chat_history_lengths: List[List[int]])->int:
	'''
	Chat history lengths are stored as a List[List[int]] eg. [[19,200],[50,234]] where 
	each inner list is the list of lengths, in tokens, of the content from system/user/assistant.
	A 'turn' refers to a turn taken by one of these roles.
	'''
	return sum(sum(a) for a in chat_history_lengths)

def endpoint(root_url: str, func: str):
	return "/".join([root_url.strip("/"),func])

async def send_receive(args, audio_stream: AudioConnection):
	# Function that wraps 3 asynchronous functions: send audio bytes to STT API, receive text from STT API, send text to LLM endpoint
	#Connect to the Assembly.ai transcription service
	URL = f'wss://api.assemblyai.com/v2/realtime/ws?sample_rate={audio_stream.audio_settings["sample_rate"]}'
	async with websockets.connect(
		URL,
		extra_headers=(("Authorization", auth_key),),
		ping_interval=5,
		ping_timeout=20
	) as _ws:

		r= await asyncio.sleep(0.1)

		session_begins = await _ws.recv() # defer until the connection to assembly ai is established.
		st_status_bar.write("I'm listening :studio_microphone:")

		async def send(args, webrtc: AudioConnection):
			while True:
				try:
					# Get minimum required amount of Audio for STT API. Note: this is blocking.
					json_data = audio_stream.processed_frames(timeout_sec=30)
					# send to STT API and await response.
					if json_data:
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
			# Get text for Assembly AI API and ship it off to the LLM endpoint.
			# Implements special word 'system prompt'. This is used as the system prompt for subsequence LLM calls (role='user')

			system_prompt=None # Sytem prompt to be used to instruct Lllama2 how to respond
			
			# Chat_history is List[List[dict]] where each dict is 1 user or agent prompt/response.
			chat_history = st.session_state[CHAT_HISTORY] # Accumulated dialogs (chat history). TODO conversation and CHAT_HISTORY are both stored in the session_state but there 
			chat_history_length = st.session_state[CHAT_HISTORY_LENGTHS] # the number of tokens in teh accumulated chat history for each turn.

			tok_counter = token_counter(args.tokenizer_model_path) # function that counts tokens in the chat_history

			sys_keywords = args.system_keywords.lower().strip()
			
			while True:
				try:
					result_str = await _ws.recv()
					if args.mode and args.mode.lower().strip()=='debug':
						st_text_output.write(result_str)
					if json.loads(result_str)['message_type']=='FinalTranscript' and json.loads(result_str)['text']!="" :
						st_user_input.write(json.loads(result_str)['text'])

						st.session_state['conversation']+=f'\n\nYou: \n {json.loads(result_str)["text"]}' # Rendered into text_area markup in head.
					
						# If 'system prompt' keyword, then store it and use it when submitting a dialog to the chat bot.
						if sys_keywords in json.loads(result_str)['text'].lower():
							system_prompt={"role":"system","content":f"{re.sub(sys_keywords,'',json.loads(result_str)['text']).strip()}"}
						else:
							st_status_bar.write('Sending to LLM...')
							if system_prompt is not None:
								# prepend the system prompt to the dialog.
								# Note that the llama2 model appears to keep track of the prior conversation ( up to the context window length - TBC)
								prompt = [system_prompt,{"role":"user","content":f"{json.loads(result_str)['text']}"}]
							else:
								prompt = [{"role":"user","content":f"{json.loads(result_str)['text']}"}]

							chat_history.append(prompt) # creates list of lists.
							chat_history_length.append([tok_counter(json.dumps(prompt))])  # will also count the json and list characters so will over estimate the length.
							
							# maintain chat history 'queue' length
							while turn_sum(chat_history_length)>0 and turn_sum(chat_history_length) > args.chat_history_length:
								chat_history=chat_history[1:] # Drop one of the dialog turns (content.)
								chat_history_length=chat_history_length[1:]

							if len(chat_history)==0:
								logger.error('Chat history was reduced to zero. This happens if the initial prompt from the user is longer than the cli max_seq_len argument.')
								continue

							logging.getLogger(__name__).info(f'Chat history length in chars @ new prompt:{turn_sum(chat_history_length)}')

							# Prepare the conversation history for sending to LLM. Llama expects List[List[dict]] (See pydantic model in chat_api.py)
							# Note this only allows for a single conversation whereas the LLM may be able to take multiple conversations at the same time.
							dialog = {"dialogs":[list(chain(*chat_history))]}
						
							# Non-blocking call to the LLM API allows voice and STT to continue running while the LLM responds.
							async with aiohttp.ClientSession() as session: #TODO make this one clientsession per instance not per request for latency reasons.
								async with session.post(url=endpoint(args.llm_endpoint,'chat'),json=dialog) as r:
									try:
										data = await r.json()
										chat_response=data['data'][0]['generation']['content']
									except KeyError as e:
										logger.error(f'Key error. Return message from endpoint: {data}')
									except Exception as e:
										logger.error(f'post error:{e.message}; LLM error message :{e["error"]}')

									st.session_state['conversation']+=f'\n\nLLM: \n {chat_response}'

									st_status_bar.write('thinking, thinking...')

							# Add response from LL to chat history.
							assistant_dialog = {"role":"assistant","content":chat_response}
							chat_history[-1].append(assistant_dialog) # prompt now consists of content from each role (system(optional),user, assistant)
							# add response to the history
							chat_history_length[-1].append(tok_counter(json.dumps(assistant_dialog)))
						
							logger.info(f'Chat history length after response:{turn_sum(chat_history_length)}')

							st.session_state[CHAT_HISTORY] =chat_history # Accumulated dialogs (chat history)
							st.session_state[CHAT_HISTORY_LENGTHS] = chat_history_length # the number of tokens in teh accumulated chat history for each turn.

							#reset system prompt as we don't need to include it for every turn of the conversation
							system_prompt = None
							st.experimental_rerun() # force postback.
						
				except websockets.exceptions.ConnectionClosedError as e:
					print(e)
					assert e.code == 4008
					break

				except Exception as e:
					assert False, "Not a websocket 4008 error"
	  
		send_result, receive_result = await asyncio.gather(send(args, audio_stream), receive(args))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_endpoint', type=str, default = 'http://localhost:8080/', help='URL for REST API serving LLM')
	parser.add_argument('--system_keywords',type=str,default='system prompt', help='phrase used to start setting of system prompt')
	parser.add_argument('--chat_history_length',type=int, default=512, help='The number tokens in the context window that available to store conversation history. Must <= maximum sequence length setting for the LLM.')
	parser.add_argument('--tokenizer_model_path', type=str, default='./tokenizer.model',help='used to calculate the tokens in the conversation history')
	parser.add_argument('--mode', type=str, choices=['quiet','debug'], default='quiet',help='debug mode exposes results from STT API call')
	parser.add_argument('--local','-l',action='store_true',default=False, help='Set this flag if no ICE server is needed.')
	parser.add_argument('--stream_type',type=str, choices=['web','local'], default='local',help='Wether audio is sources from a browser or local machine.')

	args = parser.parse_args()

	# Override come params if they are supplied by the llm_params endpoint.
	llm_params = None
	try:
		llm_params = json.loads(requests.get(endpoint(args.llm_endpoint,'llm_params')).content)
		logger.info(f"Max sequence length:{llm_params['max_seq_len']}")
		args.chat_history_length = llm_params['max_seq_len'] # makes sure that max context window in LLM is used in managing the length of the chat_history. 
	except Exception as e:
		logger.info('/llm_params endpoint in found. Default arguments from app will be used for LLM parameters.')
	
	# create the audio source connection.
	if args.stream_type=='web':
		audio_stream = WebRTCAudioSteam(None,local=args.local, timeout_sec=60, st_status_bar=st_status_bar)
	elif args.stream_type =='local':
		audio_stream = PyAudioStream(None, st_status_bar=st_status_bar) #Default are in the class - TODO refactor to config file.

	# Chat history window.
	conversation_txt=st.text_area('**Chat Window**',st.session_state['conversation'],key='conversation_txt', height=500)
	
	#UI elements
	label_col, text_col = st.columns([0.2,0.8])
	with label_col:
		st_html(st.empty(),'statusLabel','You')
	with text_col:
		st_user_input = st_html(st.empty(),'statusText',' start speaking ...')
	#render the javascript for the customer UI elements.
	st.components.v1.html(js)
	
	if audio_stream.conn is not None:
		print(f'Detected Audio Settings: {audio_stream.audio_settings}')
		st_status_bar.write('Audio connections established. Connecting to STT API...')
		asyncio.run(send_receive(args, audio_stream))