import sys
import os

# Insert the subdirectory into sys.path to find model_logic, rational_behavior, etc.
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Appendix_4/Streamlit"))
sys.path.insert(0, app_dir)

# Import the actual app module to execute it
import app
