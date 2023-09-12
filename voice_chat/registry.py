import logging 
from typing import Callable

logger = logging.getLogger(__name__)

class FuncRegistry:
	'''Class to manage parameterless callbacks'''
	def __init__(self):
		self.registry = {} # callbacks will be registered agains a client provided key. {key: Callables}
		
	def register(self,*,key:str,callback: Callable):
		if key in self.register:
			logger.info(f'{key} already in registry. Not updated.')
		else:
			self.registery[key]=callable
			
	def deregister(self,*,key):
		if key in self.registery:
			del self.registry['key']
			
	def run(self,*,key):
		self.registry[key]()
		
	def clear(self):
		self.registry.clear()
