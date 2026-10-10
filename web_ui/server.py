import os
import re
import sys
import json
import time
import threading
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

# Thread-safe training state
state_lock = threading.Lock()
current_process = None
training_state = {
    "status": "idle",  # idle, running, completed, error
    "model": "gpt2_small",
    "logs": [],
    "steps": [],
    "total_steps": 5,
    "current_step": 0,
    "latest_loss": None,
    "initial_loss": None,
    "param_delta": None,
    "elapsed_sec": 0,
    "error_message": ""
}

test_suite_state = {
    "status": "idle",
    "tests": [],
    "total_passed": 0,
    "total_tests": 19,
    "duration_sec": 0
}

MODEL_CONFIGS = {
    "gpt2_small": {
        "name": "GPT-2 Small",
        "params": "124M",
        "workdir": os.path.join(REPO_ROOT, "examples", "gpt2_small_lora_finetune"),
        "exe": os.path.join(REPO_ROOT, "examples", "gpt2_small_lora_finetune", "build", "train.exe"),
        "adapter_path": os.path.join(REPO_ROOT, "examples", "gpt2_small_lora_finetune", "outputs", "lora_smoke.safetensors"),
        "args_generator": lambda cfg: [
            "--synthetic_smoke",
            "--smoke_steps", str(cfg.get("steps", 5)),
            "--batch_size", str(cfg.get("batch_size", 4)),
            "--seq_len", str(cfg.get("seq_len", 128)),
            "--lr", str(cfg.get("lr", 0.0005)),
            "--rank", str(cfg.get("rank", 8)),
            "--alpha", str(cfg.get("alpha", 16))
        ]
    },
    "gpt2_medium": {
        "name": "GPT-2 Medium",
        "params": "355M",
        "workdir": os.path.join(REPO_ROOT, "examples", "gpt2_medium_lora_finetune"),
        "exe": os.path.join(REPO_ROOT, "examples", "gpt2_medium_lora_finetune", "build", "train.exe"),
        "adapter_path": os.path.join(REPO_ROOT, "examples", "gpt2_medium_lora_finetune", "outputs", "lora_smoke.safetensors"),
        "args_generator": lambda cfg: [
            "--synthetic_smoke",
            "--smoke_steps", str(cfg.get("steps", 5)),
            "--batch_size", str(cfg.get("batch_size", 2)),
            "--seq_len", str(cfg.get("seq_len", 128)),
            "--lr", str(cfg.get("lr", 0.0003)),
            "--rank", str(cfg.get("rank", 8)),
            "--alpha", str(cfg.get("alpha", 16))
        ]
    },
    "gemma_270m": {
        "name": "Gemma 3 (270M)",
        "params": "270M",
        "workdir": os.path.join(REPO_ROOT, "examples", "gemma_3_270m_lora_finetune"),
        "exe": os.path.join(REPO_ROOT, "examples", "gemma_3_270m_lora_finetune", "build", "train.exe"),
        "adapter_path": os.path.join(REPO_ROOT, "examples", "gemma_3_270m_lora_finetune", "gemma_lora", "gemma_lora_smoke.safetensors"),
        "args_generator": lambda cfg: [
            "--synthetic_smoke",
            "--smoke_steps", str(cfg.get("steps", 5)),
            "--lr", str(cfg.get("lr", 0.0002))
        ]
    },
    "gemma_1b_pt": {
        "name": "Gemma 3 (1B-PT)",
        "params": "1.0B",
        "workdir": os.path.join(REPO_ROOT, "examples", "gemma_3_1b_pt_lora_finetune"),
        "exe": os.path.join(REPO_ROOT, "examples", "gemma_3_1b_pt_lora_finetune", "build", "train.exe"),
        "adapter_path": os.path.join(REPO_ROOT, "examples", "gemma_3_1b_pt_lora_finetune", "gemma_lora", "gemma_lora_smoke.safetensors"),
        "args_generator": lambda cfg: [
            "--synthetic_smoke",
            "--smoke_steps", str(cfg.get("steps", 5)),
            "--lr", str(cfg.get("lr", 0.0002))
        ]
    },
    "qwen": {
        "name": "Qwen 2.5 (0.5B)",
        "params": "490M",
        "workdir": os.path.join(REPO_ROOT, "examples", "qwen_lora_finetune"),
        "exe": os.path.join(REPO_ROOT, "examples", "qwen_lora_finetune", "build", "train_wikitext.exe"),
        "adapter_path": os.path.join(REPO_ROOT, "examples", "qwen_lora_finetune", "outputs", "lora_smoke.safetensors"),
        "args_generator": lambda cfg: [
            "--synthetic_smoke",
            "--smoke_steps", str(cfg.get("steps", 5)),
            "--lr", str(cfg.get("lr", 0.0002)),
            "--lora_r", str(cfg.get("rank", 8)),
            "--lora_alpha", str(cfg.get("alpha", 16))
        ]
    }
}

def run_training_worker(model_key, config):
    global current_process, training_state
    m_info = MODEL_CONFIGS.get(model_key)
    if not m_info or not os.path.exists(m_info["exe"]):
        with state_lock:
            training_state["status"] = "error"
            training_state["error_message"] = f"Executable not found for {model_key}"
        return

    cmd = [m_info["exe"]] + m_info["args_generator"](config)
    start_time = time.time()
    step_re = re.compile(r"\[smoke step (\d+)/(\d+)\] loss=([0-9.]+) grad_norm=([0-9.]+)")
    delta_re = re.compile(r"\[SyntheticSmoke\] param_delta=([0-9.]+)")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=m_info["workdir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        with state_lock:
            current_process = proc

        for line in iter(proc.stdout.readline, ""):
            clean_line = line.rstrip()
            if not clean_line:
                continue

            with state_lock:
                training_state["logs"].append(clean_line)
                training_state["elapsed_sec"] = round(time.time() - start_time, 2)

                m_step = step_re.search(clean_line)
                if m_step:
                    step_num = int(m_step.group(1))
                    total_s = int(m_step.group(2))
                    loss_val = float(m_step.group(3))
                    grad_norm = float(m_step.group(4))

                    if training_state["initial_loss"] is None:
                        training_state["initial_loss"] = loss_val
                    training_state["latest_loss"] = loss_val
                    training_state["current_step"] = step_num
                    training_state["total_steps"] = total_s

                    training_state["steps"].append({
                        "step": step_num,
                        "total": total_s,
                        "loss": loss_val,
                        "grad_norm": grad_norm,
                        "timestamp": round(time.time() - start_time, 2)
                    })

                m_delta = delta_re.search(clean_line)
                if m_delta:
                    training_state["param_delta"] = float(m_delta.group(1))

        proc.stdout.close()
        proc.wait()

        with state_lock:
            if proc.returncode == 0:
                training_state["status"] = "completed"
            else:
                training_state["status"] = "error"
                training_state["error_message"] = f"Exited with code {proc.returncode}"
            current_process = None
    except Exception as e:
        with state_lock:
            training_state["status"] = "error"
            training_state["error_message"] = str(e)
            current_process = None

def run_tests_worker():
    global test_suite_state
    with state_lock:
        test_suite_state["status"] = "running"
        test_suite_state["tests"] = []

    build_dir = os.path.join(REPO_ROOT, "operator", "build")
    cmd = ["ctest", "--test-dir", build_dir, "--output-on-failure"]
    start_time = time.time()

    test_pattern = re.compile(r"^\s*(\d+)/19 Test\s+#(\d+):\s+(\w+)\s+\.+\s+(Passed|Failed)\s+([\d.]+)\s+sec")

    try:
        res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
        tests = []
        passed_count = 0
        for line in res.stdout.splitlines():
            m = test_pattern.search(line)
            if m:
                status = m.group(4)
                if status == "Passed":
                    passed_count += 1
                tests.append({
                    "id": int(m.group(2)),
                    "name": m.group(3),
                    "status": status,
                    "time_sec": float(m.group(5))
                })

        with state_lock:
            test_suite_state["status"] = "completed" if res.returncode == 0 else "failed"
            test_suite_state["tests"] = tests
            test_suite_state["total_passed"] = passed_count
            test_suite_state["total_tests"] = len(tests)
            test_suite_state["duration_sec"] = round(time.time() - start_time, 2)
    except Exception as e:
        with state_lock:
            test_suite_state["status"] = "error"
            test_suite_state["error_message"] = str(e)

class MobileFineTunerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with state_lock:
                models_info = {}
                for k, v in MODEL_CONFIGS.items():
                    models_info[k] = {
                        "name": v["name"],
                        "params": v["params"],
                        "exists": os.path.exists(v["exe"])
                    }
                data = {
                    "training": training_state,
                    "models": models_info,
                    "test_suite": test_suite_state
                }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif parsed.path == "/api/logs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with state_lock:
                data = {
                    "status": training_state["status"],
                    "model": training_state["model"],
                    "logs": training_state["logs"],
                    "steps": training_state["steps"],
                    "current_step": training_state["current_step"],
                    "total_steps": training_state["total_steps"],
                    "latest_loss": training_state["latest_loss"],
                    "initial_loss": training_state["initial_loss"],
                    "param_delta": training_state["param_delta"],
                    "elapsed_sec": training_state["elapsed_sec"],
                    "error_message": training_state["error_message"]
                }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif parsed.path == "/api/tests":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with state_lock:
                data = test_suite_state
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif parsed.path == "/api/artifacts":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            artifacts = []
            for k, m in MODEL_CONFIGS.items():
                p = m["adapter_path"]
                if os.path.exists(p):
                    stat = os.stat(p)
                    artifacts.append({
                        "model": m["name"],
                        "path": p,
                        "filename": os.path.basename(p),
                        "size_bytes": stat.st_size,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                    })
            self.wfile.write(json.dumps({"artifacts": artifacts}).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body)
        except Exception:
            req_data = {}

        if parsed.path == "/api/train":
            model_key = req_data.get("model", "gpt2_small")
            with state_lock:
                if training_state["status"] == "running":
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Training already in progress"}).encode("utf-8"))
                    return

                training_state["status"] = "running"
                training_state["model"] = model_key
                training_state["logs"] = []
                training_state["steps"] = []
                training_state["total_steps"] = int(req_data.get("steps", 5))
                training_state["current_step"] = 0
                training_state["latest_loss"] = None
                training_state["initial_loss"] = None
                training_state["param_delta"] = None
                training_state["elapsed_sec"] = 0
                training_state["error_message"] = ""

            t = threading.Thread(target=run_training_worker, args=(model_key, req_data), daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "model": model_key}).encode("utf-8"))
            return

        elif parsed.path == "/api/stop":
            global current_process
            with state_lock:
                if current_process:
                    try:
                        current_process.terminate()
                    except Exception:
                        pass
                    current_process = None
                training_state["status"] = "idle"
                training_state["logs"].append(">>> Training process halted by user.")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))
            return

        elif parsed.path == "/api/run-tests":
            t = threading.Thread(target=run_tests_worker, daemon=True)
            t.start()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "testing_started"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def main():
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = HTTPServer(("127.0.0.1", port), MobileFineTunerHandler)
    print(f"[*] MobileFineTuner Web UI Server listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[*] Server stopped.")

if __name__ == "__main__":
    main()
