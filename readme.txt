#Preparar entorno (solo la primera vez)

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#Configurar SUMO (cada vez que se abra una terminal)

export SUMO_HOME="/usr/share/sumo"
python traffic_generator.py
python visualize.py

#ajustar delay a 100 ms

