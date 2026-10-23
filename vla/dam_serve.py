"""DAM-3B (NVIDIA Describe Anything) as a local naming service for stack v1 (docs/stack_v1.md). It runs in its own
environment (.venv-dam: transformers 4.51; its remote code does not load under transformers 5, which the rest of the
stack uses), next to the job's other servers.

  .venv-dam/bin/python vla/dam_serve.py PORT
  POST /describe {"image": <base64 PNG>, "mask": <base64 PNG, white = region>} -> {"text": "...", "seconds": s}
Same model and settings as vla/crop_list.py's region naming (the stack report's DAM-3B numbers): prompt mode
full+focal_crop, "Describe the masked region in a few words.", temperature 0.2, top_p 0.5, 48 new tokens.
"""
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModel

Q = "<image>\nDescribe the masked region in a few words."
from huggingface_hub import snapshot_download  # noqa: E402

# the local snapshot folder, not the repo id: DAM's loader asks the Hub whether the repo exists unless given a path
# (compute nodes are offline)
path = snapshot_download("nvidia/DAM-3B-Self-Contained", local_files_only=True)
model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float16)
dam = model.to("cuda").init_dam(conv_mode="v1", prompt_mode="full+focal_crop")
lock = threading.Lock()


def png(s):
    return Image.open(io.BytesIO(base64.b64decode(s)))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # health check
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        with lock:
            out = dam.get_description(png(req["image"]).convert("RGB"), png(req["mask"]).convert("L"), Q,
                                      temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=48)
        text = out if isinstance(out, str) else "".join(out)
        body = json.dumps(dict(text=text, seconds=round(time.time() - t0, 3))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


print(f"DAM serving on {sys.argv[1]}", flush=True)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
