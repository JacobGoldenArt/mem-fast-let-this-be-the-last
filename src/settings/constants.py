import os

from dotenv import load_dotenv

load_dotenv()

USER_ID = os.getenv("JG_USER_ID")
PAYLOAD_KEY = "content"
PATH_KEY = "path"
PATCH_PATH = "user/{user_id}/core"
INSERT_PATH = "user/{user_id}/conversational/{event_id}"
TIMESTAMP_KEY = "timestamp"
TYPE_KEY = "type"
# 767 zeros followed by a small non-zero value
MINIMAL_VEC = [0.0] * 767 + [1e-6]