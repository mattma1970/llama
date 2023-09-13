from streamlit_webrtc import webrtc_streamer, WebRtcMode

from typing import List, Callable, Any, Dict
from uuid import uuid4
import streamlit as st
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
import json
import base64
import queue
import logging
import numpy as np
import av
import math

logger = logging.getLogger(__name__)

class AudioConnection(ABC):
	'''API for any audio connection e.g. webrtc or pyaudio stream.'''
	def __init__(self,conn,audio_settings={}, audio_processing_func:Callable =None, **kwargs):
		self.conn = conn  # the stream or webrtc connection..
		self.audio_settings=audio_settings #audio settings e.g. sample rate
		self.processing_func =audio_processing_func # Callable used to convert frames to format needed by consuming STT. Callable must only require a single parameter whcih must be the list of audioframes.

		self.frames = []  # raw audio frame bytes
		self.frames_available = False # flag indicating that data i

	@abstractmethod
	def ready(self):
		'''Checks if the audio connection/stream is open and ready for frames to be collected.'''
		pass
	@abstractmethod
	def _read_frames(self, timeout:float):
		'''Private method to get audio frames from the queue'''
		pass
	
	@abstractmethod
	def processed_frames(self, processing_func: Callable = None, **kwargs):
		'''Post processing function frames -> JSON required for STT API/model'''
		pass


class WebRTCAudioSteam(AudioConnection):
	'''
	webRTc connection class for Assembly.ai
	'''
	# Conditionally load streamlit_webrtc so there isn't a hard dependancy on it.
	from streamlit_webrtc import webrtc_streamer, WebRtcMode

	def __init__(self, conn: webrtc_streamer = None, audio_settings: dict = {}, audio_processing_func: Callable = None, timeout_sec:float=60.0, **kwargs):
		super().__init__(conn,audio_settings)

		if audio_processing_func is not None:
			self.processing_func = audio_processing_func
		else:
			self.processing_func = self.default_processor

		if 'max_frames' in kwargs:
			self.MAX_FRAMES= kwargs['max_frames']
		else:
			self.MAX_FRAMES=99 

		# If connection passed in then use it otherwise trigger its creation here. 
		if conn is None:
			self.conn, self.audio_settings = setup_webRTC(use_ice_server=('local' in kwargs), timeout_sec=timeout_sec, **kwargs) # This is blocking

	def default_processor(self, data: av.AudioFrame):
		'''
		Returns base64 encoded bytes from a single channel of the audioframe. Can be overridden.
		@args:
			data: av.AudioFrame : A single AudioFrame that exposes a to_ndarray function for retrieving the raw bytes data.
		@returns:
			utf-8 encoded base64 data from dataframe.
		'''
		data=np.ascontiguousarray(data.to_ndarray()[0][0::self.audio_settings['num_channels']]) # Allows for audiochannels to be interleaved
		return base64.b64encode(data).decode('utf-8')
		
	def ready(self):
		'''
			Check if the webrtc_streamer status is playing and reciever has been instantiated.
			@returns:
				bool: Indicates is streamer is playing.
		'''
		if self.conn is None:
			return False
		else:
			return self.conn.state.playing and self.conn.audio_receiver
	
	def _read_frames(self, timeout: float=60, **kwargs) -> None:
		'''Reads the audio queue and truncates it if queue was too long for STT API
		   and loads into self.frames
		'''
		self.frames = []
		if self.ready():      
			try:
				'''Ensure a minimum number of frames for sending to STT API.
					Doing this with get_frame is very slow and we quickly get queue overflow. Get_frames collect everything that is available on the queue 
					Delays after this call result in frames accumulating in the frame_queue. Its important to make sure the rate of incoming frames >= rate that the frames are consumed by the getframes function.
					To do a full fourier transform on this machine lead to a queue size of around 20 frames ( 0.4 s) Dropping the FFT reduces this to 6 frames.'''
				while len(self.frames)<self.audio_settings['required_frames']:
					self.frames.extend(self.conn.audio_receiver.get_frames(timeout=timeout)) # call to webRTC audio_reciever (airtc)
			except queue.Empty:
				logger.warning(f"Audio queue failed to receive a input from the microphone within {timeout} seconds. This could be due to network congestion, audio device or driver problems.")

			# If audioframes consumers have been blocked then make sure to limit the number of samples sent to the STT API.
			if len(self.frames) > self.MAX_FRAMES:
				self.frames=self.frames[-self.MAX_FRAMES:]
		else:
			self.frames=[]

	def processed_frames(self, **kwargs):
		if len(self.frames)==0:
			self._read_frames()

		json_data=None

		if len(self.frames)>0:
			sb = [self.processing_func(audio_frame) for audio_frame in self.frames]
			data = ''.join(sb)
			json_data = json.dumps({"audio_data":data})
			self.frames=[]
		return json_data


def setup_webRTC(use_ice_server: str = False, ice_servers: List[str]=['stun:stun.l.google.com:19302'], timeout_sec:float=60.0, st_status_bar: Any = None, **kwargs)-> (webrtc_streamer,Dict):
	'''
		Setup webRTC connection and assign it the unique identifier "unique_id".
		Blocks until connection established or timeout is reached.
		@args:
			ice_server: List[str]: list of turn or stun server URLs for webRTC routing. Defaults to free (insecure) STUN server provided by google of voip.
			timeoout: flaot: time out for connection in seconds
		@return:
			webrtc_streamer: connection object regrdless of the playing state
			dict: audio stream properties such as sample rate etc.
	'''
	audio_settings=None

	#Session_state_keys
	WEBRTC_CONNX_ESTABLISHED = 'webRTC_runtime_configuration_is_set' # Flag used  to indicate that the audio_settings of the inbound audio have been collected. This is done once when the connection is established.
	WEBRTC_CONNECTION = 'connx' # persitance of the connection and audio data from the streamer.

	if 'frames_per_buffer' in kwargs:
		FRAMES_PER_BUFFER = kwargs['frames_per_buffer']
	else:
		FRAMES_PER_BUFFER = 4800

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
		# webRTC connection where client and server are on same network. e.g when only sharing with people on the same network.
		webrtc_ctx = webrtc_streamer(
			key=unique_id,
			mode=WebRtcMode.SENDONLY, # ?? leads to instantiation of audio_reciever ??
			audio_receiver_size=1024, # buffer size in aiortc packets (20 ms of samples)
			media_stream_constraints = {"video": False,"audio":True},
			desired_playing_state=True,  # startplaying upon rendering
		)
	
	# Block until the connetion is established.	App is useless without a connectoin so blocking is justified.
	pbar = st.progress(timeout_sec)
	for i in tqdm(range(timeout_sec)): # note this gets interrupted each time there is a postback event.
		st_status_bar.write('Connecting to server. Please be patient.')
		if webrtc_ctx.state.playing:
			pbar.empty()
			break
		time.sleep(1)
		pbar.progress(i)
	
	if WEBRTC_CONNX_ESTABLISHED in st.session_state:
		# If connection has already been established, just return it and the settings.
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
		st.session_state[WEBRTC_CONNECTION]=(webrtc_ctx,audio_settings) # persist the connection and setting through post-back
		return webrtc_ctx, audio_settings
	else:
		# return placeholder object
		st_status_bar.write(f'Failed to connect to server within {timeout_sec} seconds. Please refresh page to try again.')
		return None, None
	

class PyAudioStream(AudioConnection):
	def __init__(self,conn=None, audio_settings={}, audio_processing_func:Callable=None, **kwargs):
		'''
		Class representing an audio stream and the processing tools for local audio input.
		Uses pyaudio (python bindings on portaudio)
		@args:
			conn :pyaudio.stream
			audio_settings: dict: dictionary of audio settings NOT YET IMPLEMENTED
			audio_processing_func: Callable: function to convert bytes from audio stream to format required for the TTS. Takes bytes data as the only arguament func(data)
		'''
		import pyaudio # Python bindings for portaudio
		super().__init__(conn,audio_settings)

		self.p=pyaudio.PyAudio()
		if audio_processing_func is None:
			self.processing_func = self.default_processor
		else: 
			self.processing_func = audio_processing_func

		#unlike webRTC audio settings are perscriptive. TODO move to a config file.
		self.audio_settings = {
								"required_frames":1,  # PyAudio allows a set number of samples to be read so one read is needed. Note that frame in PyAudio means sample whereas webRTC a frame is approx 0.2 s of samples.
								"sample_rate":16000,  #sample rate of incoming sound
								"num_channels":1,	  # stereo or mono. AssemblyAI requires mono.
								"frames_per_buffer":4800, # number of samples to read each pass.
								"format":pyaudio.paInt16 # format of sample data paInt16 is 16 bits per channel.
							}

		if conn is None:
			self.conn = self.open_stream()
		else:
			self.conn = conn
		
		self.frames_available=False
	
	def default_processor(self, data: bytes):
		'''
		Function to parse av.DataFrames data. This can be overidden by a passed-in audio_processing_function (hence the signature)
		'''				
		data = base64.b64encode(data).decode("utf-8")
		json_data = json.dumps({"audio_data":str(data)})
		return json_data
	
	def ready(self):
		'''Check if stream is ready to read from. PyAudio does all the hard work here. '''
		return self.conn is not None

	
	def _read_frames(self, timeout=100):
		'''Private method to fill self.frames with audio frames'''
		self.frames_available=False
		if self.conn:
			try:
				self.frames=self.conn.read(self.audio_settings['frames_per_buffer'])
				self.frames_available=True
			except:
				self.frames_available=False
		else:
			raise RuntimeError('No pyAudio stream available.')
		
	def processed_frames(self, **kwargs)->str:
			json_data=None
			if not self.frames_available:
				self._read_frames()
				json_data=self.processing_func(self.frames)
			self.frames_available = False
			self.frames=None
			return json_data
			
	def open_stream(self):
		return self.p.open(
			format=self.audio_settings['format'],
			channels=self.audio_settings['num_channels'],
			rate=self.audio_settings['sample_rate'],
			input=True,
			frames_per_buffer=self.audio_settings['frames_per_buffer']
			)
