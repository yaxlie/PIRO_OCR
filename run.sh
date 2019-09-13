#!/bin/bash
set -e
#sudo apt-get install python3-venv
python3 -m venv ./venv
chmod 755 venv/bin/activate
./venv/bin/activate
pip3 install -r ./requirements.txt
python3 main.py res/img_1.jpg
./venv/bin/deactivate
