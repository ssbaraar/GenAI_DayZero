# key_param.py

import os
from dotenv import load_dotenv

load_dotenv()

# Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# MongoDB URI
MONGO_URI = os.getenv("MONGO_URI")

# You can add other sensitive parameters here as needed