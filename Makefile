.PHONY: setup run
setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
run:
\t. .venv/bin/activate && streamlit run app.py
