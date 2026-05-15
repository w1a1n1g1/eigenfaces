import os
import subprocess

os.environ['STREAMLIT_GATHER_USAGE_STATS'] = 'false'
subprocess.run(['streamlit', 'run', 'app.py', '--server.port', '8501'])