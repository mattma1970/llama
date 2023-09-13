import streamlit as st

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
