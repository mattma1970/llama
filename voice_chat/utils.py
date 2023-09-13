'''
Misc utility functions. mattma1970@gmail
'''


import streamlit as st
from typing import List
import base64
import numpy as np
import av


def turn_sum(chat_history_lengths: List[List[int]])->int:
	'''
	Chat history lengths are stored as a List[List[int]] eg. [[19,200],[50,234]] where 
	each inner list is the list of lengths, in tokens, of the content from system/user/assistant.
	A 'turn' refers to a turn taken by one of these roles.
	'''
	return sum(sum(a) for a in chat_history_lengths)

def endpoint(root_url: str, func: str):
	return "/".join([root_url.strip("/"),func])


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
