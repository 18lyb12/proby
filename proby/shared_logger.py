from collections import deque
from datetime import datetime

class SharedLogger:
    def __init__(self, max_size=100):
        self.log_messages = deque(maxlen=max_size)

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} - {message}'
        self.log_messages.append(log_entry)

    def get_logs(self):
        return list(self.log_messages)

# Create a singleton instance of SharedLogger
shared_logger = SharedLogger()
