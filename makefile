.PHONY: run_api run_chat run_local_chat run_local_web_chat test

.DEFAULT_GOAL := run_api

MODE := quiet

run_api:
	@echo "Starting Llama 2 API"
	. .venv/bin/activate; \
	torchrun --nproc_per_node 1 voice_chat/chat_api.py --model_path llama-2-7b-chat --tokenizer_path tokenizer.model --max_batch_size 6;

run_api_debug:
	@echo "Starting Llama 2 API"
	. .venv/bin/activate; \
	torchrun --nproc_per_node 1 voice_chat/chat_api.py --model_path llama-2-7b-chat --tokenizer_path tokenizer.model --max_batch_size 6 --debug;

run_local_web_chat: voice_chat/local_web_chat_client.py
	@echo 'Start voice chat via browser - client and server co-located and so no TURN server is needed.'
	. venv/bin/activate; \
	streamlit run voice_chat/local_web_chat_client.py -- --mode $(MODE) --local

run_remote_web_chat: voice_chat/local_web_chat_client.py
	@echo 'Start voice chat via remote browser - requires the https connection; use ssl-proxy'
	. venv/bin/activate; \
	voice_chat/Ssl/ssl-proxy -from 0.0.0.0:8502 -to 0.0.0.0:8501 & \
	streamlit run voice_chat/local_web_chat_client.py --  --mode $(MODE) --stream_type web
run_local_chat:
	@echo 'Run voice chat with audio source from local machine. Uses PuAudio and requires that a sound server be running (e.g. PulseAudio )'
	. venv/bin/activate; \
	streamlit run voice_chat/local_web_chat_client.py -- --mode $(MODE) --local --stream_type local
kill_ssl:
	@echo 'stop ssl-proxy'
	ps -aux | grep ssl-proxy | kill -9 $$(awk '(NR==1){print $$2}' )

install: requirements.txt
	# ffmpeg requried from PyAv (its a pythonic binding to FFMpeg)
	sudo apt install ffmpeg; \
	pip install -r requirements.txt


