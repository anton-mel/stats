"""Node side: everything that runs on the machine under test. Takes a JobSpec,
writes records. Knows nothing about GitHub or queues."""

import os

# huggingface_hub reads these at import: its 10 s metadata/read timeouts fail whole shards on slow links
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
