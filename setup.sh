#!/bin/bash

python -m venv nyt_env
source nyt_env/bin/activate
pip install -r requirements.txt
deactivate
