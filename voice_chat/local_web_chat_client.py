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

from configure import auth_key

import requests
import re
import argparse
from typing import List, Union, Any, Callable
from uuid import uuid4
import time
from tqdm import tqdm

from registry import FuncRegistry
from webrtc_connection import connx


import logging
logger = logging.getLogger(__name__)


func_registry = FuncRegistry() # Registry for callbacks



from llama.tokenizer import Tokenizer as tok

#Globals
FRAMES_PER_BUFFER = 4800  # units = samples
MAX_FRAMES = 99 # units = Frames NOT samples. For streamlit_webrtc 1 frame=0.02 seconds. AssemblyAI, maximum duration of audio posted is 2 seconds.


#Session_state_keys
WEBRTC_CONNX_ESTABLISHED = 'webRTC_runtime_configuration_is_set' # Flag used  to indicate that the audio_settings of the inbound audio have been collected. This is done once when the connection is established.
WEBRTC_CONNECTION = 'connx' # persitance of the connection and audio data from the streamer.


#### Strealit UI ####

st.title('Sqwak: Voice chat with Llama2-7B')

# <head>
if 'conversation' not in st.session_state:
	st.session_state['conversation']=' '

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

class st_html:
	'''Extend st.write for various elements to render with unique class (css_class) for use by css selectors.'''
	def __init__(self, element, css_class: st, text: str = ' ', wrap: bool=True):
		'''
		Args:
			element: obj: If text is passed in then create a st.empty() as the element, otherwise use what was passed in.
			css_class: str: unique class name to be consumed by css selectors.
			text: str: initialization text.
			wrap: bool: indicate if the div tags should wrap the text of just be a marker.
		'''
		self.element = element
		self.css_class = css_class
		self.wrap = wrap
		self.write(text)


	@property
	def element(self):
		return self._element
	
	@element.setter
	def element(self, obj):
		if isinstance(obj,str):
			self._element=st.empty()
		else:
			self._element=obj

	def write(self, text: str):
		if self.wrap:
			text =f"<div class='{self.css_class}'>{text}</div>"
		else:
			text= f"<div class='{self.css_class} />{text}"

		self.element.write(text, unsafe_allow_html=True)
		return str
	
	def empty(self):
		self.element.empty()
		return None


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

def blocking_audio_read(webrtc: connx, timeout:int=30):
	''' Gets all audio frames on the queue. Requires at least st.session_state['required_frames'] frames before returning. 
		@args:
			webrtc: connx object with connx.conn in playing state
		@returns:
			json: audio data in json format needed for assemblyai
	'''
	if not webrtc.conn.state.playing:
		return None

	audio_frames=[]
	attempt_count =0
	if webrtc.conn.audio_receiver:      
		try:
			'''Ensure a minimum number of frames for sending to STT API.
				Doing this with get_frame is very slow and we quickly get queue overflow. Get_frames collect everything that is available on the queue 
				Delays after this call result in frames accumulating in the frame_queue. Its important to make sure the rate of incoming frames >= rate that the frames are consumed by the getframes function.
				To do a full fourier transform on this machine lead to a queue size of around 20 frames ( 0.4 s) Dropping the FFT reduces this to 6 frames.'''
			while len(audio_frames)<webrtc.audio_settings['required_frames']:
				audio_frames.extend(webrtc.conn.audio_receiver.get_frames(timeout=timeout))
		except queue.Empty:
			logger.warning(f"Audio queue failed to receive a input from the microphone within {timeout} seconds. This could be due to network congestion, audio device or driver problems.")

	# If audioframes consumers have been blocked then make sure to limit the number of samples sent to the STT API.
	if len(audio_frames) > MAX_FRAMES:
		audio_frames=audio_frames[-MAX_FRAMES:]

	# extract the bytes from the audioframes and encode them to base64 a requried by the STT API.
	sb = [stt_b64_encode(audio_frame, webrtc.audio_settings['num_channels']) for audio_frame in audio_frames]
	data = ''.join(sb)
	json_data = json.dumps({"audio_data":data})

	return json_data

async def async_animated_status(message: str, wait_duration_ms: int = 200, end_condition: Callable = None):
	'''Keep adding .. to the message until the end condition is true'''
	while True:
		st_status_bar.write(message)
		if end_condition():
			break
		r=await asyncio.sleep(wait_duration_ms/1000)
		if end_condition():
			break
		message+='.'
		st_status_bar.write(message)
		r=await asyncio.sleep(wait_duration_ms/1000)


async def send_receive(args, webrtc: connx):
	# Function that wraps 3 asynchronous functions: send audio bytes to STT API, receive text from STT API, send text to LLM endpoint
	#Connect to the Assembly.ai transcription service
	URL = f'wss://api.assemblyai.com/v2/realtime/ws?sample_rate={webrtc.audio_settings["sample_rate"]}'
	async with websockets.connect(
		URL,
		extra_headers=(("Authorization", auth_key),),
		ping_interval=5,
		ping_timeout=20
	) as _ws:

		r= await asyncio.sleep(0.1)

		session_begins = await _ws.recv() # defer until the connection to assembly ai is established.
		st_status_bar.write("I'm listening :studio_microphone:")

		async def send(args, webrtc: connx):
			while True:
				try:
					# Get minimum required amount of Audio for STT API. Note: this is blocking.
					json_data = blocking_audio_read(webrtc,30)
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
			
			chat_history = [] # Accumulated conversation.
			chat_history_length = [] # the number of tokens in teh accumulated chat history for each turn.
			tok_counter = token_counter(args.tokenizer_model_path) # functino that counts tokens in the chat_history

			sys_keywords = args.system_keywords.lower().strip()
			
			while True:
				try:
					result_str = await _ws.recv()
					if args.mode and args.mode.lower().strip()=='debug':
						st_text_output.write(result_str)
					if json.loads(result_str)['message_type']=='FinalTranscript' and json.loads(result_str)['text']!="" :
						st_user_input.write(json.loads(result_str)['text'])

						st.session_state['conversation']+=f'\nYou: \n{json.loads(result_str)["text"]}'
						#conversation_txt.value=st.session_state['conversation']

						
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
							
							# track the conversation history
							""" if chat_history:
								chat_history.extend(prompt)
								chat_history_length.append(tok_counter(prompt))
							else:
								chat_history_length=[tok_counter(prompt)]
								chat_history = prompt

							# maintain chat history 'queue' length
							while sum(chat_history_length)>args.chat_history_length:
								chat_history=chat_history[1:]
								chat_history_length=chat_history_length[1:]														
							"""

							async with aiohttp.ClientSession() as session: #TODO make this one clientsession ber instance not per request for latency reasons.
								async with session.post(url=args.llm_endpoint,json={"dialogs":prompt}) as r:
									data = await r.json()
									try:
										chat_response=data['data'][0]['generation']['content']
									except Exception as e:
										logger.error(f'post error:{e.message}; LLM error message :{e["error"]}')

									st.session_state['conversation']+=f'\nLLM: \n {chat_response}'
									#conversation_txt.value=st.session_state['conversation']
									st_status_bar.write('..your turn')

							# add response to the history
							chat_history.extend([{"role":"assistant","content":chat_response}])

							#reset system prompt as we don't need to include it for every turn of the conversation
							system_prompt = None
							st.experimental_rerun()
						
				except websockets.exceptions.ConnectionClosedError as e:
					print(e)
					assert e.code == 4008
					break

				except Exception as e:
					assert False, "Not a websocket 4008 error"
	  
		send_result, receive_result = await asyncio.gather(send(args, webrtc), receive(args))


def setup_webRTC(use_ice_server: str = False, ice_servers: List[str]=['stun:stun.l.google.com:19302'], timeout_sec:float=60.0)-> connx:
	'''
		Setup webRTC connection and assign it the unique identifier "unique_id".
		Blocks until connection established or timeout is reached.
		@args:
			ice_server: List[str]: list of turn or stun server URLs for webRTC routing. Defaults to free (insecure) STUN server provided by google of voip.
			timeoout: flaot: time out for connection in seconds
		@return:
			webrtc_streamer: connection object regrdless of the playing state
	'''
	audio_settings=None
	""" 	if 'conn_progress' not in func_registry.registry:
		func_registry.register(finish)
		r=await init() """
	try:
		unique_id = st.session_state['sess_unique_id']
	except:		
		unique_id = str(uuid4())[:10]
		st.session_state['sess_unique_id'] = unique_id # persist it over the post backs

	if use_ice_server:
		# webRTc connection via a TURN server
		webrtc_ctx = webrtc_streamer(
			key=unique_id,
			mode=WebRtcMode.SENDONLY, # ?? leads to instantiation of audio_reciever ??
			audio_receiver_size=1024, # buffer size in aiortc packets (20 ms of samples)
			rtc_configuration={
				"iceServers": [{"urls": ice_servers}]},
			media_stream_constraints = {"video": False,"audio":True},
			desired_playing_state=True,  # startplaying upon rendering
		)
	else:
		# webRTC connection where client and server are on same network.
		webrtc_ctx = webrtc_streamer(
			key=unique_id,
			mode=WebRtcMode.SENDONLY, # ?? leads to instantiation of audio_reciever ??
			audio_receiver_size=1024, # buffer size in aiortc packets (20 ms of samples)
			media_stream_constraints = {"video": False,"audio":True},
			desired_playing_state=True,  # startplaying upon rendering
		)
	
	# Block until the connetion is established.
	pbar = st.progress(timeout_sec)

	for i in tqdm(range(timeout_sec)): # note this gets interrupted each time there is a postback
		st_status_bar.write('Connecting to server')
		if webrtc_ctx.state.playing:
			pbar.empty()
			break
		time.sleep(1)
		pbar.progress(i)
	
	if WEBRTC_CONNX_ESTABLISHED in st.session_state:
		# If connection has already been established, just return it.
		return st.session_state[WEBRTC_CONNECTION]
	elif webrtc_ctx.state.playing and WEBRTC_CONNX_ESTABLISHED not in st.session_state:
		# Collect details on the inbound audio frames to use in audio processing. 
		# In aiortc a frame consists of 20ms of samples. Depending on the sample rate the number of samples will vary. 
		# Clears the current frames queue
		first_packet = webrtc_ctx.audio_receiver.get_frames(timeout=1)[0]
		audio_settings = {
							"required_frames":math.ceil(float(FRAMES_PER_BUFFER/first_packet.samples)),  #min frames required for AssemblyAi API. A good choice is 4800 samples
							"sample_rate":first_packet.sample_rate,										 #sample rate of incoming sound
							"num_channels":len(first_packet.layout.channels)							 # stereo or mono. AssemblyAI requires mono.
							}
		st.session_state[WEBRTC_CONNX_ESTABLISHED]=True # Flag that the settings have been collected.
		webrtc_conn = connx(webrtc_ctx, audio_settings)
		st.session_state[WEBRTC_CONNECTION]=webrtc_conn # Flag that the settings have been collected.
		return webrtc_conn
	else:
		# return placeholder object
		st_status_bar.write(f'Failed to connect to server within {timeout_sec} seconds. Please refresh page to try again.')
		return connx(None,None)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_endpoint', type=str, default = 'http://localhost:8080/chat', help='URL for REST API serving LLM')
	parser.add_argument('--system_keywords',type=str,default='system prompt', help='phrase used to start setting of system prompt')
	parser.add_argument('--chat_history_length',type=int, default=3000, help='The number tokens in the context window that available to store conversation history')
	parser.add_argument('--tokenizer_model_path', type=str, default='./tokenizer.model',help='used to calculate the tokens in the conversation history')
	parser.add_argument('--mode', type=str, default='debug')
	parser.add_argument('--local','-l',action='store_true',default=False, help='Set this flag if no ICE server is needed.')

	args = parser.parse_args()

	webrtc_conn = setup_webRTC(use_ice_server=args.local, timeout_sec=60) # This is blocking
		# Chat history window.
	conversation_txt=st.text_area('**Chat Window**',st.session_state['conversation'],key='conversation_txt', height=500)
	
	#UI elements
	label_col, text_col = st.columns([0.2,0.8])
	with label_col:
		st_html(st.empty(),'statusLabel','You')
	with text_col:
		st_user_input = st_html(st.empty(),'statusText',' start speaking ...')
	
	st.components.v1.html(js)
	
	if webrtc_conn.conn is not None:
		print(f'Detected Audio Settings: {webrtc_conn.audio_settings}')
		st_status_bar.write('webRTC connection established. Connecting to STT API...')
		asyncio.run(send_receive(args, webrtc_conn))