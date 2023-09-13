'''
A facade class for managing the chat history via streamlits state_session.
Psuedo code:
    monolog = dict{'role':... ,'content':...}
    turn = [monolog] <- at most, one for each role.
    dialog = [turns] <- List[List[monologs]]

Chats consist of a list of turns or monologs where each entry is the content provided by one of the defined roles ( user/system/assistant)
'''

import streamlit as st
from typing import List, Callable
import json
import logging
from utils import turn_sum

logger = logging.getLogger('chatHistory')

CHAT_HISTORY = 'chat_dialogs' # list of list of dialogs to submit to LLM i.e. dialog history.
CHAT_HISTORY_LENGTHS = 'chat_history'


class history:
    '''Session_state backed class for chat history'''
    def __init__(self, token_counter: Callable, max_history: int):

        self.max_history=max_history    # maximum length of history in tokens. Should match the context window of the LLM.
        self.token_counter = token_counter  #LLM specific token counter.

        if CHAT_HISTORY not in st.session_state:
            st.session_state[CHAT_HISTORY]=[]
            st.session_state[CHAT_HISTORY_LENGTHS]=[]

    @property
    def chat_history(self):
        return st.session_state[CHAT_HISTORY]
    
    @chat_history.setter
    def chat_history(self, hist: List[List[dict]]):
        st.session_state[CHAT_HISTORY]=hist

    @property
    def chat_history_lengths(self):
        return st.session_state[CHAT_HISTORY_LENGTHS]
    
    @chat_history_lengths.setter
    def chat_history_lengths(self, lengths: List[List[int]]):
        st.session_state[CHAT_HISTORY_LENGTHS]=lengths

    def get_history(self):
        return self.chat_history, self.chat_history_lengths
    
    def add(self, prompt: List[dict]):
        ''' Add a new list of system,user monologs'''
        self.chat_history.append(prompt)
        self.chat_history_lengths.append([self.token_counter(json.dumps(prompt))])

    def extend_last(self, response: List[dict]):
        ''' Add a monolog to the last monolog in the list. Typically from the assistant.'''
        self.chat_history[-1].extend(response)
        self.chat_history_lengths[-1].append(self.token_counter(json.dumps(response)))
        logger.info(f'Chat history length after response:{turn_sum(self.chat_history_lengths)}')

    def calc_history_length(self):
        return turn_sum(self.chat_history_lengths) # calc sum of values in list of list.
    
    def truncate_history(self):
        # maintain chat history 'queue' length but trimming until the turn lenghts are below the limit
        while turn_sum(self.chat_history_lengths)>0 and turn_sum(self.chat_history_lengths) > self.max_history:
            self.chat_history=self.chat_history[1:] # Drop one of the dialog turns (content.)
            self.chat_history_lengths=self.chat_history_lengths[1:]
        if len(self.chat_history)==0:
            logger.error('Chat history was reduced to zero. This happens if the initial prompt from the user is longer than the cli max_seq_len argument.')
        
        logger.info(f'Chat history length in chars @ new prompt:{turn_sum(self.chat_history_lengths)}')
        return None
    


