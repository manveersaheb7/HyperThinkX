import os
import sys
import urllib.request
import re
import json
import time
import queue
import threading
import subprocess
import webbrowser
import speech_recognition as sr
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLineEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# DEPENDENCY MANAGER
class DependencyManager:
    def __init__(self):
        self.ddgs = None
        self.bs4 = None
        self.qdrant = None
        self.qdrant_models = None
        self.requests = None
        self._load()

    def _load(self):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS
        except ImportError:
            pass
        try:
            import requests
            self.requests = requests
        except ImportError:
            pass
        try:
            from bs4 import BeautifulSoup
            self.bs4 = BeautifulSoup
        except ImportError:
            pass
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, Distance, VectorParams
            self.qdrant = QdrantClient
            self.qdrant_models = (PointStruct, Distance, VectorParams)
        except ImportError:
            pass

DEPS = DependencyManager()

# CONFIGURATION
class Configuration:
    def __init__(self):
        self.MODEL_CHAT = os.environ.get("HYPERTHINK_X_CHAT_MODEL", "gemma3:4b")
        self.MODEL_EMBED = "nomic-embed-text"
        self.OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.OLLAMA_TIMEOUT = 120.0
        self.DEBUG_MODE = True

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        self.MEM_PATH = os.path.join(self.BASE_DIR, "hyperthink_x_memory.json")
        self.VECTOR_PATH = os.path.join(self.BASE_DIR, "hyperthink_vectors")

        self.QUICK_DEADLINE = 30.0
        self.DEEP_DEADLINE = 180.0
        self.COST_INPUT = 0.50
        self.COST_OUTPUT = 1.50
        self.WAKE_WORDS = ["jarvis", "hello", "hyper", "brother"]
        self.MAX_CHAT_HISTORY = 20
        self.MAX_RETRIES = 2
        self.RETRY_DELAY = 1.0

CONFIG = Configuration()

SYSTEM_PROMPT = """
You are HyperThink X — a Production-Grade Deep Research Agent. Version - 1.0

You dynamically specialize based on the user’s query into one of three domains:

1) Technical & Coding Research (Engineering focus)
2) E-Commerce Intelligence (Business/Product focus)
3) Financial & Market Analysis (Investment/Strategy focus)

You must first classify the query domain before answering.

------------------------------------------------------------
DOMAIN BEHAVIOR RULES
------------------------------------------------------------

If query relates to:
- engineering, AI, coding, architecture, RAG, systems, ML, APIs, research papers
→ Operate as Senior Technical Research Engineer.

If query relates to:
- products, pricing, reviews, SKUs, competitors, margins, GMV, CAC, marketplaces
→ Operate as E-Commerce Intelligence Analyst.

If query relates to:
- earnings, EBITDA, ROE, valuation, stock comparison, bull/bear case, macro risk
→ Operate as Financial Research Analyst.

------------------------------------------------------------
CORE CAPABILITIES (ALL DOMAINS)
------------------------------------------------------------

1. Dual Research Modes:
   - Quick Mode (<30s): High-signal concise structured insight.
   - Deep Mode (<3min): Multi-source synthesis with structured reasoning.

2. Persistent Memory:
   - Remember user preferences.
   - Remember domain-specific KPIs.
   - Use memory to improve future responses.

3. Interactive Flow:
   - Ask clarifying questions when ambiguous.
   - Support follow-ups ("go deeper", "compare", "optimize instead").

4. Production Readiness:
   - Structured output.
   - Clear citations.
   - Confidence estimation.
   - State assumptions explicitly.
   - Handle missing data gracefully.

------------------------------------------------------------
DOMAIN-SPECIFIC MEMORY RULES
------------------------------------------------------------

Technical Domain:
- Remember preferred depth (code examples, architecture diagrams, math detail).

E-Commerce Domain:
- Remember preferred KPIs (GMV, CAC, LTV, margins).
- Remember preferred marketplace.
- Remember category interest.

Financial Domain:
- Remember risk tolerance (conservative/aggressive).
- Remember preferred KPIs (EBITDA, ROE, FCF).
- Remember sector/geography interest.

------------------------------------------------------------
RESEARCH OUTPUT FORMAT (MANDATORY)
------------------------------------------------------------

## Executive Summary
[2–3 sentence overview]

## Key Findings / Approaches
[Numbered list]

## Comparative Analysis
[Pros/cons, tradeoffs, benchmarking where relevant]

## Assumptions & Production Considerations
[State assumptions clearly. Mention constraints.]

## Risks / Uncertainty Factors
[Data gaps, contradictions, limitations]

## Citations
[Numbered references]

## Confidence Level
[HIGH / MEDIUM / LOW — short justification]

------------------------------------------------------------

If user input is conversational and not research-related,
respond naturally and concisely.

Always respond in English.
"""

# GLOBAL STATE
class GlobalState:
    def __init__(self):
        self.in_queue = queue.Queue(maxsize=200)
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_speaking = threading.Event()
        self.mem_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.recognizer = sr.Recognizer()
        self.agent_state = {
            "pending": None,
            "chat_mode": False,
            "awaiting_command": False,
            "wake_timestamp": 0,
            "manual_listen": False,
            "awaiting_clarification": False,
            "clarification_context": None
        }
        self.last_stats = {"latency": 0.0, "tokens": 0, "cost": 0.0}

STATE = GlobalState()

# SIGNALS
class Communicator(QObject):
    text_signal = pyqtSignal(str, str)
    stream_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(str)
    thinking_signal = pyqtSignal(bool)

COMM = Communicator()

# LOGGER
class Logger:
    @staticmethod
    def log(message, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        if CONFIG.DEBUG_MODE:
            print(f"[{ts}] [{level}] {message}")

    @staticmethod
    def error(message):
        Logger.log(message, "ERROR")

# VECTOR DB (Qdrant)
class VectorDBManager:
    def __init__(self):
        self.client = None
        self.collection_name = "research_knowledge"
        self._connect()

    def _connect(self):
        if DEPS.qdrant:
            try:
                self.client = DEPS.qdrant(path=CONFIG.VECTOR_PATH)
                self._ensure_collection()
                Logger.log("Vector DB Connected.")
            except Exception as e:
                Logger.error(f"Vector DB Init Failed: {e}")

    def _ensure_collection(self):
        if self.client:
            try:
                self.client.get_collection(self.collection_name)
            except:
                PointStruct, Distance, VectorParams = DEPS.qdrant_models
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

    def embed(self, text):
        try:
            req = urllib.request.Request(
                f"{CONFIG.OLLAMA_HOST}/api/embeddings",
                data=json.dumps({"model": CONFIG.MODEL_EMBED, "prompt": text}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())["embedding"]
        except Exception as e:
            Logger.error(f"Embedding failed: {e}")
            return None

    def insert(self, query, answer, metadata=None):
        if not self.client:
            return
        vector = self.embed(query)
        if vector:
            PointStruct = DEPS.qdrant_models[0]
            payload = {"query": query, "answer": answer[:8000], "timestamp": time.time()}
            if metadata:
                payload.update(metadata)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=int(time.time() * 1000), vector=vector, payload=payload)]
            )

    def query(self, text, threshold=0.82):
        if not self.client:
            return None
        vector = self.embed(text)
        if vector:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector, limit=3
                )
                if hits and hits[0].score > threshold:
                    return hits[0].payload
            except Exception as e:
                Logger.error(f"Vector query failed: {e}")
        return None

    def query_multiple(self, text, limit=3, threshold=0.7):
        if not self.client:
            return []
        vector = self.embed(text)
        if vector:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector, limit=limit
                )
                return [h.payload for h in hits if h.score > threshold]
            except Exception as e:
                Logger.error(f"Vector multi-query failed: {e}")
        return []

VECTOR_DB = VectorDBManager()

# JSON MEMORY
class JSONMemoryManager:
    def __init__(self):
        self.data = {
            "profile": {"name": "User", "preferences": []},
            "chat_turns": [],
            "research_history": []
        }
        self.load()

    def load(self):
        with STATE.mem_lock:
            if os.path.exists(CONFIG.MEM_PATH):
                try:
                    with open(CONFIG.MEM_PATH, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        for key in self.data:
                            if key in loaded:
                                self.data[key] = loaded[key]
                        migrated = []
                        for turn in self.data["chat_turns"]:
                            if isinstance(turn, dict):
                                migrated.append({
                                    "ts": turn.get("ts", 0),
                                    "u": turn.get("u", turn.get("user", "")),
                                    "a": turn.get("a", turn.get("assistant", ""))
                                })
                        self.data["chat_turns"] = migrated
                    Logger.log(f"Memory loaded: {len(self.data['chat_turns'])} turns")
                except Exception as e:
                    Logger.error(f"Memory Load Error: {e}")
                    self.data = {
                        "profile": {"name": "User", "preferences": []},
                        "chat_turns": [],
                        "research_history": []
                    }

    def save(self):
        with STATE.mem_lock:
            try:
                with open(CONFIG.MEM_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2)
            except Exception as e:
                Logger.error(f"Memory Save Error: {e}")

    def add_turn(self, user, assistant):
        self.data["chat_turns"].append({
            "ts": time.time(),
            "u": str(user) if user else "",
            "a": str(assistant) if assistant else ""
        })
        self.data["chat_turns"] = self.data["chat_turns"][-15:]
        self.save()

    def add_research_record(self, record):
        self.data["research_history"].append(record)
        self.data["research_history"] = self.data["research_history"][-60:]
        self.save()

    def get_last_research(self):
        records = self.data.get("research_history", [])
        return records[-1] if records else None

    def update_preference(self, text):
        match = re.search(r"\b(remember|i prefer|i like|i want|always use)\s+(.*)", text.strip(), re.IGNORECASE)
        if match:
            pref = match.group(2).strip()
            if pref and pref not in self.data["profile"]["preferences"]:
                self.data["profile"]["preferences"].append(pref)
                self.data["profile"]["preferences"] = self.data["profile"]["preferences"][-20:]
                self.save()
                return f'Got it! I\'ll remember: "{pref}". This will improve my future answers for you.'
        return None

    def get_context_string(self):
        name = self.data["profile"].get("name", "User")
        prefs = self.data["profile"].get("preferences", [])
        if prefs:
            return f"User: {name} | Preferences: {'; '.join(prefs)}"
        return f"User: {name}"

    def get_chat_history_messages(self, limit=None):
        if limit is None:
            limit = CONFIG.MAX_CHAT_HISTORY
        recent = self.data["chat_turns"][-limit:]
        messages = []
        for turn in recent:
            if not isinstance(turn, dict):
                continue
            user_msg = turn.get("u", "") or turn.get("user", "") or ""
            asst_msg = turn.get("a", "") or turn.get("assistant", "") or ""
            if not user_msg:
                continue
            messages.append({"role": "user", "content": str(user_msg)})
            if asst_msg:
                ans = str(asst_msg)
                if len(ans) > 1500:
                    ans = ans[:1500] + "...[truncated]"
                messages.append({"role": "assistant", "content": ans})
        return messages

    def get_preference_prompt_section(self):
        prefs = self.data["profile"].get("preferences", [])
        if not prefs:
            return ""
        return "\n\nUSER PREFERENCES (apply these to your response):\n" + "\n".join(f"- {p}" for p in prefs)
MEMORY = JSONMemoryManager()

# TTS MANAGER
class TTSManager:
    _current_process = None  # Track the running TTS process
    _process_lock = threading.Lock()

    @staticmethod
    def stop_speaking():
        """Immediately kill the TTS process and clear the queue."""
        with TTSManager._process_lock:
            if TTSManager._current_process and TTSManager._current_process.poll() is None:
                try:
                    # Windows specific process tree killing
                    if sys.platform.startswith("win"):
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(TTSManager._current_process.pid)], 
                                     capture_output=True, creationflags=0x08000000)
                    else:
                        TTSManager._current_process.kill()
                    
                    TTSManager._current_process.wait(timeout=1)
                    Logger.log("TTS System Interrupted and Cleaned.")
                except Exception as e:
                    Logger.error(f"TTS kill error: {e}")
                finally:
                    TTSManager._current_process = None

        # Clear the queue so it doesn't start the next sentence
        while not STATE.tts_queue.empty():
            try:
                STATE.tts_queue.get_nowait()
            except queue.Empty:
                break

        STATE.is_speaking.clear()
        COMM.status_signal.emit("Ready")

    @staticmethod
    def speak_system(text):
        """Core speaking logic with watchdog timer."""
        clean_text = text.strip()
        if not clean_text:
            return
            
        # Optimization: Limit long responses to prevent TTS freeze
        if len(clean_text) > 400:
            clean_text = clean_text[:400] + "... Check chat for the full technical breakdown."

        COMM.status_signal.emit("Speaking...")
        try:
            STATE.is_speaking.set()
            
            if sys.platform.startswith("win"):
                # Escape double quotes for PowerShell
                safe_text = clean_text.replace('"', "'").replace("\n", " ")
                ps_cmd = (
                    'Add-Type -AssemblyName System.Speech; '
                    '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$s.Speak("{safe_text}");'
                )
                
                with TTSManager._process_lock:
                    TTSManager._current_process = subprocess.Popen(
                        ["powershell", "-NoProfile", "-Command", ps_cmd],
                        creationflags=0x08000000  # CREATE_NO_WINDOW
                    )
                
                # Watchdog: Don't let a single TTS process hang for more than 20 seconds
                try:
                    TTSManager._current_process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    TTSManager.stop_speaking()

            elif sys.platform == "darwin":
                with TTSManager._process_lock:
                    TTSManager._current_process = subprocess.Popen(["say", clean_text])
                TTSManager._current_process.wait()
            else:
                with TTSManager._process_lock:
                    TTSManager._current_process = subprocess.Popen(["espeak", clean_text])
                TTSManager._current_process.wait()

        except Exception as e:
            Logger.error(f"TTS Error: {e}")
        finally:
            with TTSManager._process_lock:
                TTSManager._current_process = None
            time.sleep(0.2) # Small buffer to prevent overlap
            STATE.is_speaking.clear()
            COMM.status_signal.emit("Ready")

    @staticmethod
    def worker():
        """Background thread to process speech queue."""
        while not STATE.stop_event.is_set():
            try:
                text = STATE.tts_queue.get(timeout=0.5)
                TTSManager.speak_system(text)
            except queue.Empty:
                continue
            except Exception as e:
                Logger.error(f"TTS Worker Error: {e}")

    @staticmethod
    def enqueue(text):
        if text:
            STATE.tts_queue.put(text)

# --- WEB SEARCH & SCRAPING: WITH DEADLINE MANAGEMENT ---
class WebSearchManager:
    @staticmethod
    def search_with_retry(query, max_results, retries=None):
        if retries is None:
            retries = CONFIG.MAX_RETRIES
        if not DEPS.ddgs:
            return []
            
        for attempt in range(retries + 1):
            try:
                with DEPS.ddgs() as ddgs:
                    # Support for different DDGS versions
                    try:
                        return list(ddgs.text(query, max_results=max_results))
                    except TypeError:
                        return list(ddgs.text(query))
            except Exception as e:
                Logger.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    time.sleep(CONFIG.RETRY_DELAY)
        return []

    @staticmethod
    def scrape(url, timeout=6.0, char_limit=5000):
        if not (DEPS.requests and DEPS.bs4):
            return ""
        try:
            # Using headers to avoid bot detection
            resp = DEPS.requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if 200 <= resp.status_code < 300:
                soup = DEPS.bs4(resp.text, "html.parser")
                # Remove junk to save tokens
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                return " ".join(soup.get_text().split())[:char_limit]
        except Exception as e:
            Logger.error(f"Scrape failed for {url}: {e}")
        return ""

    @staticmethod
    def gather_intel(query, mode, deadline):
        """Scrapes web data but respects the time deadline."""
        start_time = time.time()
        limit = 3 if mode == "quick" else 6
        
        raw_results = WebSearchManager.search_with_retry(query, limit)
        processed = []
        
        for i, res in enumerate(raw_results, 1):
            # If we are nearing the deadline, stop scraping more sites
            if time.time() - start_time > deadline * 0.8:
                Logger.log("Web Research deadline reached. Finalizing current data.")
                break
                
            entry = {
                "id": i, "title": res.get("title", ""),
                "url": res.get("href", ""), "snippet": res.get("body", "")
            }
            
            if mode == "deep":
                COMM.status_signal.emit(f"Deep scanning [{i}/{limit}]...")
                full_text = WebSearchManager.scrape(entry["url"])
                if full_text:
                    entry["content"] = full_text
                    
            processed.append(entry)
            
        return processed

    @staticmethod
    def format_sources(sources):
        if not sources:
            return "(No external sources found)"
        formatted = []
        for s in sources:
            block = f"[{s['id']}] {s['title']} ({s['url']})\nSummary: {s['snippet']}"
            if s.get("content"):
                block += f"\nContent: {s['content'][:800]}..." # Token limit optimization
            formatted.append(block)
        return "\n\n".join(formatted)

    @staticmethod
    def compute_source_confidence(sources):
        if not sources: return "LOW", "No valid sources"
        count = len(sources)
        has_deep = any(s.get("content") for s in sources)
        
        if count >= 4 and has_deep: return "HIGH", f"{count} sources + deep analysis"
        if count >= 2: return "MEDIUM", f"{count} sources indexed"
        return "LOW", f"Limited data ({count} source)"

# --- WEB SEARCH & SCRAPING: WITH DEADLINE MANAGEMENT ---
class WebSearchManager:
    @staticmethod
    def search_with_retry(query, max_results, retries=None):
        if retries is None:
            retries = CONFIG.MAX_RETRIES
        if not DEPS.ddgs:
            return []
            
        for attempt in range(retries + 1):
            try:
                with DEPS.ddgs() as ddgs:
                    # Support for different DDGS versions
                    try:
                        return list(ddgs.text(query, max_results=max_results))
                    except TypeError:
                        return list(ddgs.text(query))
            except Exception as e:
                Logger.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    time.sleep(CONFIG.RETRY_DELAY)
        return []

    @staticmethod
    def scrape(url, timeout=6.0, char_limit=5000):
        if not (DEPS.requests and DEPS.bs4):
            return ""
        try:
            # Using headers to avoid bot detection
            resp = DEPS.requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if 200 <= resp.status_code < 300:
                soup = DEPS.bs4(resp.text, "html.parser")
                # Remove junk to save tokens
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                return " ".join(soup.get_text().split())[:char_limit]
        except Exception as e:
            Logger.error(f"Scrape failed for {url}: {e}")
        return ""

    @staticmethod
    def gather_intel(query, mode, deadline):
        """Scrapes web data but respects the time deadline."""
        start_time = time.time()
        limit = 3 if mode == "quick" else 6
        
        raw_results = WebSearchManager.search_with_retry(query, limit)
        processed = []
        
        for i, res in enumerate(raw_results, 1):
            # If we are nearing the deadline, stop scraping more sites
            if time.time() - start_time > deadline * 0.8:
                Logger.log("Web Research deadline reached. Finalizing current data.")
                break
                
            entry = {
                "id": i, "title": res.get("title", ""),
                "url": res.get("href", ""), "snippet": res.get("body", "")
            }
            
            if mode == "deep":
                COMM.status_signal.emit(f"Deep scanning [{i}/{limit}]...")
                full_text = WebSearchManager.scrape(entry["url"])
                if full_text:
                    entry["content"] = full_text
                    
            processed.append(entry)
            
        return processed

    @staticmethod
    def format_sources(sources):
        if not sources:
            return "(No external sources found)"
        formatted = []
        for s in sources:
            block = f"[{s['id']}] {s['title']} ({s['url']})\nSummary: {s['snippet']}"
            if s.get("content"):
                block += f"\nContent: {s['content'][:800]}..." # Token limit optimization
            formatted.append(block)
        return "\n\n".join(formatted)

    @staticmethod
    def compute_source_confidence(sources):
        if not sources: return "LOW", "No valid sources"
        count = len(sources)
        has_deep = any(s.get("content") for s in sources)
        
        if count >= 4 and has_deep: return "HIGH", f"{count} sources + deep analysis"
        if count >= 2: return "MEDIUM", f"{count} sources indexed"
        return "LOW", f"Limited data ({count} source)"
# LLM ENGINE
class LLMEngine:
    @staticmethod
    def count_tokens(text):
        return max(1, len(text or "") // 4)

    @staticmethod
    def stream_response(messages, sys_prompt, deadline=None):
        start_time = time.time()
        accumulated = ""
        for attempt in range(CONFIG.MAX_RETRIES + 1):
            try:
                if deadline and (time.time() - start_time) > deadline * 0.9:
                    break
                COMM.thinking_signal.emit(False)
                COMM.text_signal.emit("", "ai_start")

                full_messages = [{"role": "system", "content": sys_prompt}] + messages
                payload = json.dumps({
                    "model": CONFIG.MODEL_CHAT,
                    "messages": full_messages,
                    "stream": True
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{CONFIG.OLLAMA_HOST}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )

                timeout = CONFIG.OLLAMA_TIMEOUT
                if deadline:
                    timeout = min(timeout, max(10, deadline - (time.time() - start_time)))

                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    for chunk in resp:
                        if not chunk:
                            continue
                        data = json.loads(chunk.decode("utf-8"))
                        token = data.get("message", {}).get("content", "")
                        if token:
                            accumulated += token
                            COMM.stream_signal.emit(token)
                        if data.get("done"):
                            break
                        if deadline and (time.time() - start_time) > deadline:
                            accumulated += "\n\n[Truncated — deadline reached]"
                            COMM.stream_signal.emit("\n\n[Truncated — deadline reached]")
                            break
                break
            except Exception as e:
                Logger.error(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < CONFIG.MAX_RETRIES:
                    time.sleep(CONFIG.RETRY_DELAY)
                else:
                    COMM.text_signal.emit(f"\n[LLM Failed: {e}]", "system")
                    return ""

        duration = time.time() - start_time
        in_tok = LLMEngine.count_tokens(str(messages))
        out_tok = LLMEngine.count_tokens(accumulated)
        cost = ((in_tok * CONFIG.COST_INPUT) + (out_tok * CONFIG.COST_OUTPUT)) / 1_000_000
        COMM.stats_signal.emit(f"Latency: {duration:.2f}s | Tokens: {in_tok}+{out_tok} | Cost: ${cost:.6f}")

        with STATE.stats_lock:
            STATE.last_stats = {"latency": duration, "tokens": in_tok + out_tok, "cost": cost}

        return accumulated.strip()

# EXECUTION MANAGER
class ExecutionManager:
    @staticmethod
    def open_url(url):
        if isinstance(url, str):
            webbrowser.open(url.strip())

    @staticmethod
    def run_cmd(args):
        if isinstance(args, list):
            subprocess.Popen(args, shell=True)

    @staticmethod
    def extract_code_block(text):
        pattern = re.compile(r"<CODE>(.*?)</CODE>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"^```python\n?", "", code, flags=re.IGNORECASE)
            code = re.sub(r"^```\n?", "", code)
            code = re.sub(r"\n?```$", "", code)
            return code, True
        return text.strip(), False

    @staticmethod
    def execute_safe(code):
        try:
            sandbox = {
                "open_url": ExecutionManager.open_url,
                "run_process": ExecutionManager.run_cmd,
                "print": print, "len": len, "range": range, "str": str, "int": int
            }
            exec(code, sandbox)
            return True, "Execution Successful"
        except Exception as e:
            return False, str(e)

# CLARIFICATION ENGINE
class ClarificationEngine:
    VAGUE_PATTERNS = [
        (r"^(explain|tell me about|what is|what are)\s+\w+$",
         "That's a broad topic. Would you like a **quick overview** or a **deep technical analysis** with code examples and tradeoffs?"),
        (r"^compare\s+",
         "Sure! Could you specify:\n1. What aspects to compare? (performance, cost, ease of use)\n2. Quick summary or deep analysis?"),
        (r"^(how to|how do i)\s+",
         "Do you want:\n1. A quick step-by-step guide\n2. A deep analysis with multiple approaches and tradeoffs?"),
        (r"^(improve|optimize|fix)\s+",
         "Could you clarify:\n1. What's the current problem?\n2. Any constraints (budget, time, technology)?"),
    ]

    @staticmethod
    def needs_clarification(query):
        q = query.strip().lower()
        words = q.split()
        if len(words) <= 2 and not any(x in q for x in ["open", "quit", "hello", "hi", "hey"]):
            return True, f'Your query "{query}" is quite brief. Could you provide more context?\n- What specific aspect?\n- Quick answer or deep research?'
        for pattern, question in ClarificationEngine.VAGUE_PATTERNS:
            if re.match(pattern, q):
                return True, question
        return False, None

    @staticmethod
    def is_followup(query):
        q = query.lower()
        indicators = ["go deeper", "more detail", "elaborate", "previous", "earlier",
                       "last search", "expand on", "what about", "tell me more",
                       "its specs", "its price", "about it", "that one"]
        return any(ind in q for ind in indicators)

# RESEARCH AGENT
class ResearchAgent:
    @staticmethod
    def parse_mode(text):
        t = text.lower()
        mode = "quick"
        if any(x in t for x in ["hyper think", "hyperthink", "deep dive", "deep search",
                                  "detailed", "report", "deep analysis", "in-depth"]):
            mode = "deep"
        clean = re.sub(r"(deep:|quick:|deep search|deep dive|hyper think|hyperthink|deep analysis|in-depth)",
                       "", text, flags=re.IGNORECASE).strip()
        return mode, clean

    @staticmethod
    def is_research_query(text):
        t = text.lower()
        triggers = [
            "search", "find", "analyze", "research", "deep", "hyperthink",
            "compare", "benchmark", "tradeoffs", "trade-offs", "approaches",
            "overview of", "dive into", "explore", "investigate",
            "what are the best", "pros and cons", "state of the art", "latest in", "report on"
        ]
        return any(re.search(trigger, t) for trigger in triggers)

    @staticmethod
    def execute_research(mode, query):
        research_start = time.time()
        deadline = CONFIG.QUICK_DEADLINE if mode == "quick" else CONFIG.DEEP_DEADLINE

        if mode == "quick":
            cache = VECTOR_DB.query(query)
            if cache:
                COMM.text_signal.emit("[Memory Hit — cached research]", "system")
                COMM.text_signal.emit("", "ai_start")
                COMM.stream_signal.emit(cache["answer"])
                COMM.stats_signal.emit("Source: Vector Cache | Cost: $0.00")
                return cache["answer"]

        related = VECTOR_DB.query_multiple(query, limit=2, threshold=0.65)
        related_context = ""
        if related:
            related_context = "\n\nRELATED PREVIOUS RESEARCH:\n"
            for r in related:
                related_context += f"- Q: {r.get('query', '')}\n  A: {r.get('answer', '')[:500]}...\n"

        COMM.status_signal.emit(f"Researching ({mode.upper()})...")
        sources = WebSearchManager.gather_intel(query, mode, deadline)
        source_text = WebSearchManager.format_sources(sources)
        confidence_level, confidence_reason = WebSearchManager.compute_source_confidence(sources)

        mem_ctx = MEMORY.get_context_string()
        pref_section = MEMORY.get_preference_prompt_section()
        history_msgs = MEMORY.get_chat_history_messages(limit=6)

        research_prompt = (
            f"RESEARCH TASK ({mode.upper()} MODE)\n"
            f"User Context: {mem_ctx}\n{pref_section}\n"
            f"Query: {query}\n{related_context}\n"
            f"Web Sources:\n{source_text}\n\n"
            f"Source Confidence: {confidence_level} — {confidence_reason}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Use the structured format (Executive Summary, Key Approaches, Tradeoffs, Production Considerations, Citations, Confidence Level).\n"
            f"2. Cite sources using [1], [2], etc.\n"
            f"3. Confidence Level: {confidence_level} — {confidence_reason}\n"
            f"4. Note any contradictions between sources.\n"
            f"5. Apply user preferences if any.\n"
        )

        messages = history_msgs + [{"role": "user", "content": research_prompt}]
        remaining = deadline - (time.time() - research_start)
        COMM.status_signal.emit("Synthesizing...")
        response = LLMEngine.stream_response(messages, SYSTEM_PROMPT, deadline=remaining)

        if not response:
            response = ResearchAgent._fallback(sources, query, confidence_level)
            COMM.text_signal.emit("", "ai_start")
            COMM.stream_signal.emit(response)

        total_time = time.time() - research_start
        MEMORY.add_research_record({
            "ts": time.time(), "mode": mode, "query": query,
            "answer": response[:5000], "confidence": confidence_level,
            "latency": round(total_time, 2), "source_count": len(sources),
            "sources": [{"n": s["id"], "t": s["title"], "u": s["url"]} for s in sources]
        })
        VECTOR_DB.insert(query, response, metadata={"mode": mode, "confidence": confidence_level})
        return response

    @staticmethod
    def _fallback(sources, query, confidence):
        parts = [f"## Partial Results for: {query}\n", "*LLM synthesis failed. Raw source summaries:*\n"]
        for s in sources:
            parts.append(f"**[{s['id']}] {s['title']}**\nURL: {s['url']}\nSummary: {s['snippet']}\n")
        parts.append(f"\n**Confidence: {confidence}** — Raw data only.")
        return "\n".join(parts)

# INPUT PROCESSOR
class InputProcessor:
    @staticmethod
    def handle_text(text, source):
        cmd = text.strip()
        if not cmd:
            return

        try:
            TTSManager.stop_speaking()
            if source == "mic" and not STATE.agent_state["chat_mode"]:
                active_window = 10.0
                is_active = False
                if STATE.agent_state["manual_listen"]:
                    is_active = True
                    STATE.agent_state["manual_listen"] = False
                elif STATE.agent_state["awaiting_command"]:
                    if time.time() - STATE.agent_state["wake_timestamp"] < active_window:
                        is_active = True
                    else:
                        STATE.agent_state["awaiting_command"] = False

                found_wake = False
                for w in CONFIG.WAKE_WORDS:
                    if w in cmd.lower():
                        found_wake = True
                        cmd = re.sub(r"\b" + re.escape(w) + r"\b", "", cmd, flags=re.IGNORECASE).strip()
                        break

                if not is_active and not found_wake:
                    return
                if found_wake and not cmd:
                    STATE.agent_state["awaiting_command"] = True
                    STATE.agent_state["wake_timestamp"] = time.time()
                    COMM.text_signal.emit("[Listening...]", "system")
                    TTSManager.enqueue("Yes?")
                    return
                STATE.agent_state["awaiting_command"] = False

            COMM.text_signal.emit(f"{cmd}", "user")
            cl = cmd.lower()

            # ── Quick shortcuts ──
            # Browsers & Websites
            if "open youtube" in cl:
                TTSManager.enqueue("Opening YouTube"); ExecutionManager.open_url("https://youtube.com"); MEMORY.add_turn(cmd, "Opening YouTube"); return
            if "open google" in cl:
                TTSManager.enqueue("Opening Google"); ExecutionManager.open_url("https://google.com"); MEMORY.add_turn(cmd, "Opening Google"); return
            if "open gmail" in cl:
                TTSManager.enqueue("Opening Gmail"); ExecutionManager.open_url("https://mail.google.com"); MEMORY.add_turn(cmd, "Opening Gmail"); return
            if "open github" in cl:
                TTSManager.enqueue("Opening GitHub"); ExecutionManager.open_url("https://github.com"); MEMORY.add_turn(cmd, "Opening GitHub"); return
            if "open chatgpt" in cl:
                TTSManager.enqueue("Opening ChatGPT"); ExecutionManager.open_url("https://chat.openai.com"); MEMORY.add_turn(cmd, "Opening ChatGPT"); return
            if "open whatsapp" in cl:
                TTSManager.enqueue("Opening WhatsApp"); ExecutionManager.open_url("https://web.whatsapp.com"); MEMORY.add_turn(cmd, "Opening WhatsApp"); return
            if "open instagram" in cl:
                TTSManager.enqueue("Opening Instagram"); ExecutionManager.open_url("https://instagram.com"); MEMORY.add_turn(cmd, "Opening Instagram"); return
            if "open twitter" in cl or "open x" in cl:
                TTSManager.enqueue("Opening Twitter"); ExecutionManager.open_url("https://x.com"); MEMORY.add_turn(cmd, "Opening Twitter"); return
            if "open facebook" in cl:
                TTSManager.enqueue("Opening Facebook"); ExecutionManager.open_url("https://facebook.com"); MEMORY.add_turn(cmd, "Opening Facebook"); return
            if "open linkedin" in cl:
                TTSManager.enqueue("Opening LinkedIn"); ExecutionManager.open_url("https://linkedin.com"); MEMORY.add_turn(cmd, "Opening LinkedIn"); return
            if "open reddit" in cl:
                TTSManager.enqueue("Opening Reddit"); ExecutionManager.open_url("https://reddit.com"); MEMORY.add_turn(cmd, "Opening Reddit"); return
            if "open spotify" in cl:
                TTSManager.enqueue("Opening Spotify"); ExecutionManager.open_url("https://open.spotify.com"); MEMORY.add_turn(cmd, "Opening Spotify"); return
            if "open netflix" in cl:
                TTSManager.enqueue("Opening Netflix"); ExecutionManager.open_url("https://netflix.com"); MEMORY.add_turn(cmd, "Opening Netflix"); return
            if "open amazon" in cl:
                TTSManager.enqueue("Opening Amazon"); ExecutionManager.open_url("https://amazon.in"); MEMORY.add_turn(cmd, "Opening Amazon"); return
            if "open flipkart" in cl:
                TTSManager.enqueue("Opening Flipkart"); ExecutionManager.open_url("https://flipkart.com"); MEMORY.add_turn(cmd, "Opening Flipkart"); return
            if "open stackoverflow" in cl or "open stack overflow" in cl:
                TTSManager.enqueue("Opening Stack Overflow"); ExecutionManager.open_url("https://stackoverflow.com"); MEMORY.add_turn(cmd, "Opening Stack Overflow"); return
            if "open maps" in cl or "open google maps" in cl:
                TTSManager.enqueue("Opening Google Maps"); ExecutionManager.open_url("https://maps.google.com"); MEMORY.add_turn(cmd, "Opening Google Maps"); return
            if "open drive" in cl or "open google drive" in cl:
                TTSManager.enqueue("Opening Google Drive"); ExecutionManager.open_url("https://drive.google.com"); MEMORY.add_turn(cmd, "Opening Google Drive"); return
            if "open docs" in cl or "open google docs" in cl:
                TTSManager.enqueue("Opening Google Docs"); ExecutionManager.open_url("https://docs.google.com"); MEMORY.add_turn(cmd, "Opening Google Docs"); return
            if "open sheets" in cl or "open google sheets" in cl:
                TTSManager.enqueue("Opening Google Sheets"); ExecutionManager.open_url("https://sheets.google.com"); MEMORY.add_turn(cmd, "Opening Google Sheets"); return
            if "open telegram" in cl:
                TTSManager.enqueue("Opening Telegram"); ExecutionManager.open_url("https://web.telegram.org"); MEMORY.add_turn(cmd, "Opening Telegram"); return
            if "open discord" in cl:
                TTSManager.enqueue("Opening Discord"); ExecutionManager.open_url("https://discord.com/app"); MEMORY.add_turn(cmd, "Opening Discord"); return
            if "open pinterest" in cl:
                TTSManager.enqueue("Opening Pinterest"); ExecutionManager.open_url("https://pinterest.com"); MEMORY.add_turn(cmd, "Opening Pinterest"); return
            if "open quora" in cl:
                TTSManager.enqueue("Opening Quora"); ExecutionManager.open_url("https://quora.com"); MEMORY.add_turn(cmd, "Opening Quora"); return

            # Windows Apps
            if "open calculator" in cl or "open calc" in cl:
                TTSManager.enqueue("Opening Calculator"); ExecutionManager.run_cmd(["calc"]); MEMORY.add_turn(cmd, "Opening Calculator"); return
            if "open notepad" in cl:
                TTSManager.enqueue("Opening Notepad"); ExecutionManager.run_cmd(["notepad"]); MEMORY.add_turn(cmd, "Opening Notepad"); return
            if "open paint" in cl:
                TTSManager.enqueue("Opening Paint"); ExecutionManager.run_cmd(["mspaint"]); MEMORY.add_turn(cmd, "Opening Paint"); return
            if "open cmd" in cl or "open command prompt" in cl or "open terminal" in cl:
                TTSManager.enqueue("Opening Command Prompt"); ExecutionManager.run_cmd(["cmd"]); MEMORY.add_turn(cmd, "Opening CMD"); return
            if "open powershell" in cl:
                TTSManager.enqueue("Opening PowerShell"); ExecutionManager.run_cmd(["powershell"]); MEMORY.add_turn(cmd, "Opening PowerShell"); return
            if "open file explorer" in cl or "open explorer" in cl or "open files" in cl:
                TTSManager.enqueue("Opening File Explorer"); ExecutionManager.run_cmd(["explorer"]); MEMORY.add_turn(cmd, "Opening File Explorer"); return
            if "open settings" in cl:
                TTSManager.enqueue("Opening Settings"); ExecutionManager.run_cmd(["start", "ms-settings:"]); MEMORY.add_turn(cmd, "Opening Settings"); return
            if "open task manager" in cl:
                TTSManager.enqueue("Opening Task Manager"); ExecutionManager.run_cmd(["taskmgr"]); MEMORY.add_turn(cmd, "Opening Task Manager"); return
            if "open control panel" in cl:
                TTSManager.enqueue("Opening Control Panel"); ExecutionManager.run_cmd(["control"]); MEMORY.add_turn(cmd, "Opening Control Panel"); return
            if "open word" in cl or "open microsoft word" in cl:
                TTSManager.enqueue("Opening Word"); ExecutionManager.run_cmd(["start", "winword"]); MEMORY.add_turn(cmd, "Opening Word"); return
            if "open excel" in cl:
                TTSManager.enqueue("Opening Excel"); ExecutionManager.run_cmd(["start", "excel"]); MEMORY.add_turn(cmd, "Opening Excel"); return
            if "open powerpoint" in cl or "open ppt" in cl:
                TTSManager.enqueue("Opening PowerPoint"); ExecutionManager.run_cmd(["start", "powerpnt"]); MEMORY.add_turn(cmd, "Opening PowerPoint"); return
            if "open snipping tool" in cl or "open screenshot" in cl:
                TTSManager.enqueue("Opening Snipping Tool"); ExecutionManager.run_cmd(["snippingtool"]); MEMORY.add_turn(cmd, "Opening Snipping Tool"); return
            if "open camera" in cl:
                TTSManager.enqueue("Opening Camera"); ExecutionManager.run_cmd(["start", "microsoft.windows.camera:"]); MEMORY.add_turn(cmd, "Opening Camera"); return
            if "open clock" in cl or "open alarm" in cl:
                TTSManager.enqueue("Opening Clock"); ExecutionManager.run_cmd(["start", "ms-clock:"]); MEMORY.add_turn(cmd, "Opening Clock"); return
            if "open store" in cl or "open microsoft store" in cl:
                TTSManager.enqueue("Opening Microsoft Store"); ExecutionManager.run_cmd(["start", "ms-windows-store:"]); MEMORY.add_turn(cmd, "Opening Store"); return
            if "open photos" in cl:
                TTSManager.enqueue("Opening Photos"); ExecutionManager.run_cmd(["start", "ms-photos:"]); MEMORY.add_turn(cmd, "Opening Photos"); return
            if "open mail" in cl or "open outlook" in cl:
                TTSManager.enqueue("Opening Mail"); ExecutionManager.run_cmd(["start", "outlookmail:"]); MEMORY.add_turn(cmd, "Opening Mail"); return
            if "open calendar" in cl:
                TTSManager.enqueue("Opening Calendar"); ExecutionManager.run_cmd(["start", "outlookcal:"]); MEMORY.add_turn(cmd, "Opening Calendar"); return
            if "open weather" in cl:
                TTSManager.enqueue("Opening Weather"); ExecutionManager.run_cmd(["start", "bingweather:"]); MEMORY.add_turn(cmd, "Opening Weather"); return
            if "open maps app" in cl:
                TTSManager.enqueue("Opening Maps"); ExecutionManager.run_cmd(["start", "bingmaps:"]); MEMORY.add_turn(cmd, "Opening Maps App"); return
            if "open recorder" in cl or "open voice recorder" in cl:
                TTSManager.enqueue("Opening Voice Recorder"); ExecutionManager.run_cmd(["start", "ms-screenclip:"]); MEMORY.add_turn(cmd, "Opening Recorder"); return
            if "open sticky notes" in cl:
                TTSManager.enqueue("Opening Sticky Notes"); ExecutionManager.run_cmd(["start", "ms-stickynotes:"]); MEMORY.add_turn(cmd, "Opening Sticky Notes"); return

            # Dev Tools
            if "open vs code" in cl or "open vscode" in cl or "open visual studio code" in cl:
                TTSManager.enqueue("Opening VS Code"); ExecutionManager.run_cmd(["code"]); MEMORY.add_turn(cmd, "Opening VS Code"); return
            if "open android studio" in cl:
                TTSManager.enqueue("Opening Android Studio"); ExecutionManager.run_cmd(["start", "studio64"]); MEMORY.add_turn(cmd, "Opening Android Studio"); return
            if "open pycharm" in cl:
                TTSManager.enqueue("Opening PyCharm"); ExecutionManager.run_cmd(["start", "pycharm64"]); MEMORY.add_turn(cmd, "Opening PyCharm"); return
            if "open intellij" in cl:
                TTSManager.enqueue("Opening IntelliJ"); ExecutionManager.run_cmd(["start", "idea64"]); MEMORY.add_turn(cmd, "Opening IntelliJ"); return
            if "open postman" in cl:
                TTSManager.enqueue("Opening Postman"); ExecutionManager.run_cmd(["start", "postman"]); MEMORY.add_turn(cmd, "Opening Postman"); return
            if "open docker" in cl:
                TTSManager.enqueue("Opening Docker"); ExecutionManager.run_cmd(["start", "docker"]); MEMORY.add_turn(cmd, "Opening Docker"); return
            if "open git bash" in cl:
                TTSManager.enqueue("Opening Git Bash"); ExecutionManager.run_cmd(["start", "git-bash"]); MEMORY.add_turn(cmd, "Opening Git Bash"); return

            # System Commands
            if "shutdown" in cl or "shut down" in cl:
                TTSManager.enqueue("Shutting down in 30 seconds. Type shutdown abort to cancel."); ExecutionManager.run_cmd(["shutdown", "/s", "/t", "30"]); MEMORY.add_turn(cmd, "Shutdown initiated"); return
            if "restart" in cl or "reboot" in cl:
                TTSManager.enqueue("Restarting in 30 seconds."); ExecutionManager.run_cmd(["shutdown", "/r", "/t", "30"]); MEMORY.add_turn(cmd, "Restart initiated"); return
            if "shutdown abort" in cl or "cancel shutdown" in cl:
                TTSManager.enqueue("Shutdown cancelled."); ExecutionManager.run_cmd(["shutdown", "/a"]); MEMORY.add_turn(cmd, "Shutdown cancelled"); return
            if "lock screen" in cl or "lock computer" in cl or "lock pc" in cl:
                TTSManager.enqueue("Locking screen."); ExecutionManager.run_cmd(["rundll32.exe", "user32.dll,LockWorkStation"]); MEMORY.add_turn(cmd, "Screen locked"); return
            if "sleep" in cl and ("computer" in cl or "pc" in cl or "system" in cl):
                TTSManager.enqueue("Putting to sleep."); ExecutionManager.run_cmd(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"]); MEMORY.add_turn(cmd, "Sleep mode"); return

            # Volume (Windows)
            if "mute" in cl or "unmute" in cl:
                TTSManager.enqueue("Toggling mute.")
                ps = 'Add-Type -TypeDefinition @"\nusing System.Runtime.InteropServices;\npublic class Audio {\n[DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, int dwExtraInfo);\n}\n"@\n[Audio]::keybd_event(0xAD,0,0,0); [Audio]::keybd_event(0xAD,0,2,0)'
                ExecutionManager.run_cmd(["powershell", "-NoProfile", "-Command", ps])
                MEMORY.add_turn(cmd, "Mute toggled"); return

            # Quick Search (opens Google search)
            if cl.startswith("search ") or cl.startswith("google "):
                search_query = cmd[7:].strip() if cl.startswith("search ") else cmd[7:].strip()
                if search_query:
                    TTSManager.enqueue(f"Searching for {search_query}")
                    ExecutionManager.open_url(f"https://www.google.com/search?q={search_query.replace(' ', '+')}")
                    MEMORY.add_turn(cmd, f"Searched: {search_query}")
                    return

            # YouTube Search
            if cl.startswith("play ") or "youtube search" in cl:
                search_query = cmd[5:].strip() if cl.startswith("play ") else cmd.replace("youtube search", "").strip()
                if search_query:
                    TTSManager.enqueue(f"Searching YouTube for {search_query}")
                    ExecutionManager.open_url(f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}")
                    MEMORY.add_turn(cmd, f"YouTube search: {search_query}")
                    return

            if "what time" in cl or "current time" in cl or "time" == cl:
                now = datetime.now().strftime("%I:%M %p")
                TTSManager.enqueue(f"The time is {now}")
                COMM.text_signal.emit("", "ai_start")
                COMM.stream_signal.emit(f"The current time is **{now}**")
                MEMORY.add_turn(cmd, f"Time: {now}")
                COMM.status_signal.emit("Ready")
                return
            if "what date" in cl or "today's date" in cl or "date" == cl:
                today = datetime.now().strftime("%B %d, %Y")
                TTSManager.enqueue(f"Today is {today}")
                COMM.text_signal.emit("", "ai_start")
                COMM.stream_signal.emit(f"Today's date is **{today}**")
                MEMORY.add_turn(cmd, f"Date: {today}")
                COMM.status_signal.emit("Ready")
                return
            if "what day" in cl:
                day = datetime.now().strftime("%A")
                TTSManager.enqueue(f"Today is {day}")
                COMM.text_signal.emit("", "ai_start")
                COMM.stream_signal.emit(f"Today is **{day}**")
                MEMORY.add_turn(cmd, f"Day: {day}")
                COMM.status_signal.emit("Ready")
                return

            # Preference saving
            mem_resp = MEMORY.update_preference(cmd)
            if mem_resp:
                COMM.thinking_signal.emit(False)
                COMM.text_signal.emit("", "ai_start")
                COMM.stream_signal.emit(mem_resp)
                TTSManager.enqueue(mem_resp)
                MEMORY.add_turn(cmd, mem_resp)
                COMM.status_signal.emit("Ready")
                return

            # Handle clarification response
            if STATE.agent_state.get("awaiting_clarification") and STATE.agent_state.get("clarification_context"):
                ctx = STATE.agent_state["clarification_context"]
                STATE.agent_state["awaiting_clarification"] = False
                STATE.agent_state["clarification_context"] = None
                enriched = f"{ctx.get('original_query', '')} — User clarified: {cmd}"
                mode = ctx.get("mode", "quick")
                if any(x in cl for x in ["deep", "detailed", "in-depth", "full"]):
                    mode = "deep"
                elif any(x in cl for x in ["quick", "brief", "short", "overview"]):
                    mode = "quick"
                response = ResearchAgent.execute_research(mode, enriched)
                COMM.thinking_signal.emit(False)
                if response:
                    TTSManager.enqueue(response)
                MEMORY.add_turn(cmd, response or "")
                COMM.status_signal.emit("Ready")
                return

            # Follow-up detection
            if ClarificationEngine.is_followup(cmd):
                last = MEMORY.get_last_research()
                if last:
                    prev_q = last.get("query", "")
                    prev_a = str(last.get("answer", ""))[:500]
                    enriched = f"{cmd} (Previous context: {prev_q} — {prev_a})"
                    mode, _ = ResearchAgent.parse_mode(cmd)
                    response = ResearchAgent.execute_research(mode, enriched)
                    COMM.thinking_signal.emit(False)
                    if response:
                        TTSManager.enqueue(response)
                    MEMORY.add_turn(cmd, response or "")
                    COMM.status_signal.emit("Ready")
                    return

            # Research vs conversation
            is_research = ResearchAgent.is_research_query(cmd)

            if is_research:
                mode, query = ResearchAgent.parse_mode(cmd)
                needs_clar, clar_q = ClarificationEngine.needs_clarification(query)
                if needs_clar:
                    STATE.agent_state["awaiting_clarification"] = True
                    STATE.agent_state["clarification_context"] = {"original_query": query, "mode": mode}
                    COMM.thinking_signal.emit(False)
                    COMM.text_signal.emit("", "ai_start")
                    COMM.stream_signal.emit(clar_q)
                    TTSManager.enqueue(clar_q)
                    MEMORY.add_turn(cmd, clar_q)
                    COMM.status_signal.emit("Awaiting Clarification")
                    return
                response = ResearchAgent.execute_research(mode, query)
            else:
                history_msgs = MEMORY.get_chat_history_messages()
                pref_section = MEMORY.get_preference_prompt_section()
                history_msgs.append({"role": "user", "content": str(cmd)})
                response = LLMEngine.stream_response(history_msgs, SYSTEM_PROMPT + pref_section)

            COMM.thinking_signal.emit(False)

            if not response:
                TTSManager.enqueue("I couldn't process that. Please try again.")
                COMM.status_signal.emit("Ready")
                return

            code_content, is_code = ExecutionManager.extract_code_block(response)
            if is_code:
                ok, msg = ExecutionManager.execute_safe(code_content)
                TTSManager.enqueue("Done." if ok else f"Failed: {msg}")
                COMM.text_signal.emit(f"System: {msg}", "system")
            else:
                TTSManager.enqueue(response)

            MEMORY.add_turn(cmd, response)
            COMM.status_signal.emit("Ready")

        except Exception as e:
            Logger.error(f"handle_text error: {e}")
            COMM.text_signal.emit(f"Error: {e}", "system")
            COMM.thinking_signal.emit(False)
            COMM.status_signal.emit("Ready")

    @staticmethod
    def worker():
        while not STATE.stop_event.is_set():
            try:
                item = STATE.in_queue.get(timeout=0.2)
                if isinstance(item, tuple) and len(item) == 2:
                    source, text = item
                else:
                    source, text = "gui", str(item)
                InputProcessor.handle_text(text, source)
            except queue.Empty:
                continue
            except Exception as e:
                Logger.error(f"InputProcessor Error: {e}")
                COMM.text_signal.emit(f"Process Error: {e}", "system")
                COMM.thinking_signal.emit(False)
                COMM.status_signal.emit("Ready")

# MICROPHONE
class MicrophoneManager:
    @staticmethod
    def worker():
        try:
            with sr.Microphone() as source:
                STATE.recognizer.adjust_for_ambient_noise(source, duration=1)
                Logger.log("Microphone ready.")
                while not STATE.stop_event.is_set():
                    if STATE.is_speaking.is_set():
                        time.sleep(0.5)
                        continue
                    try:
                        if STATE.agent_state["manual_listen"] or STATE.agent_state["awaiting_command"]:
                            COMM.status_signal.emit("Listening...")
                        audio = STATE.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                        if STATE.is_speaking.is_set():
                            continue
                        COMM.status_signal.emit("Processing Voice...")
                        text = STATE.recognizer.recognize_google(audio).lower()
                        if text and len(text) > 1:
                            STATE.in_queue.put(("mic", text))
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        pass
                    except Exception as e:
                        Logger.error(f"Mic Error: {e}")
                        time.sleep(1)
        except Exception as e:
            COMM.text_signal.emit(f"Microphone Init Failed: {e}", "system")

# GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperThink X — 1.0")
        self.resize(1100, 800)
        
        # Remove default Windows title bar
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background-color: #0B141A;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Custom Title Bar ──
        self.title_bar = QFrame()
        self.title_bar.setFixedHeight(40)
        self.title_bar.setStyleSheet("background-color: #202C33;")
        
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(20, 0, 10, 0)
        
        # App name
        title_label = QLabel("HyperThink X - 1.0")
        title_label.setStyleSheet("color: #00FFCC; font-size: 16px; font-weight: bold;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Window control buttons
        btn_style = """
            QPushButton {
                background-color: transparent; color: white;
                border: none; font-size: 18px; padding: 5px 15px;
            }
            QPushButton:hover { background-color: #2A3942; }
        """
        
        minimize_btn = QPushButton("─")
        minimize_btn.setFixedSize(50, 40)
        minimize_btn.setStyleSheet(btn_style)
        minimize_btn.clicked.connect(self.showMinimized)
        minimize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        maximize_btn = QPushButton("❐")
        maximize_btn.setFixedSize(50, 40)
        maximize_btn.setStyleSheet(btn_style)
        maximize_btn.clicked.connect(self.toggle_maximize)
        maximize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(50, 40)
        close_btn.setStyleSheet(btn_style.replace("2A3942", "E81123"))
        close_btn.clicked.connect(self.close)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        title_layout.addWidget(minimize_btn)
        title_layout.addWidget(maximize_btn)
        title_layout.addWidget(close_btn)
        
        layout.addWidget(self.title_bar)
        
        # For dragging window
        self.dragging = False
        self.offset = None

        # ── Chat Area ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: #0B141A;")
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()
        self.scroll_area.setWidget(self.chat_container)
        layout.addWidget(self.scroll_area)

        # ── Stats Bar ──
        stats_container = QWidget()
        stats_container.setFixedHeight(50)
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(5, 2, 5, 2)
        stats_layout.setSpacing(2)

        self.info_label = QLabel("Latency: 0s | Tokens: 0 | Cost: $0.00")
        self.info_label.setStyleSheet("color: #8696A0; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.info_label)

        self.status_label = QLabel("SYSTEM ONLINE")
        self.status_label.setStyleSheet("color: #00FFCC; font-weight: bold; font-size: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_layout.addWidget(self.status_label)

        layout.addWidget(stats_container)

        # ── Input Area ──
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #202C33;")
        input_frame.setFixedHeight(80)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(15, 10, 15, 10)
        input_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message... (use 'deep:' for deep research)")
        self.input_field.setStyleSheet("""
            QLineEdit {
                color: white; border: none; font-size: 18px;
                background-color: #2A3942; border-radius: 15px; padding: 8px 15px;
            }
        """)
        self.input_field.returnPressed.connect(self.on_send)
        input_layout.addWidget(self.input_field)

        self.mic_button = QPushButton("🎙️")
        self.mic_button.setFixedSize(50, 50)
        self.mic_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #00A884; color: white;
                border-radius: 25px; font-size: 20px; border: none;
            }
            QPushButton:hover { background-color: #06cf9c; }
            QPushButton:pressed { background-color: #05a37b; }
        """)
        self.mic_button.clicked.connect(self.on_mic)
        input_layout.addWidget(self.mic_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(input_frame)

        # ── Connect Signals ──
        COMM.text_signal.connect(self.add_message_bubble)
        COMM.stream_signal.connect(self.update_stream)
        COMM.status_signal.connect(self.status_label.setText)
        COMM.stats_signal.connect(self.info_label.setText)
        COMM.thinking_signal.connect(self.toggle_thinking)
        
        self.current_stream_bubble = None

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    # Window dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 80:
            self.dragging = True
            self.offset = event.position().toPoint()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.offset:
            self.move(self.pos() + event.position().toPoint() - self.offset)
    
    def mouseReleaseEvent(self, event):
        self.dragging = False

    def on_send(self):
        text = self.input_field.text().strip()
        if text:
            if text.lower() == "quit":
                self.close()
            else:
                STATE.in_queue.put(("gui", text))
            self.input_field.clear()

    def on_mic(self):
        STATE.agent_state["manual_listen"] = True
        self.status_label.setText("Listening...")
        TTSManager.enqueue("Listening.")

    def toggle_thinking(self, active):
        if active:
            self.status_label.setText("THINKING...")
        else:
            self.status_label.setText("ONLINE")

    def add_message_bubble(self, text, role):
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setFont(QFont("Segoe UI", 11))
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        h_layout = QHBoxLayout()

        if role == "user":
            bubble.setStyleSheet("""
                background-color: #005C4B; color: white; padding: 12px;
                border-radius: 12px; border-bottom-right-radius: 0px;
            """)
            h_layout.addStretch()
            h_layout.addWidget(bubble)
        elif role == "ai_start":
            self.current_stream_bubble = QLabel("")
            self.current_stream_bubble.setWordWrap(True)
            self.current_stream_bubble.setFont(QFont("Segoe UI", 11))
            self.current_stream_bubble.setStyleSheet("""
                background-color: #202C33; color: white; padding: 12px;
                border: 1px solid #333; border-radius: 12px; 
                border-bottom-left-radius: 0px;
            """)
            self.current_stream_bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            h_layout.addWidget(self.current_stream_bubble)
            h_layout.addStretch()
            self.chat_layout.addLayout(h_layout)
            self.scroll_to_bottom()
            return
        elif role == "system":
            bubble.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
            bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
            h_layout.addWidget(bubble)

        self.chat_layout.addLayout(h_layout)
        self.scroll_to_bottom()

    def update_stream(self, text):
        if self.current_stream_bubble:
            self.current_stream_bubble.setText(self.current_stream_bubble.text() + text)
            self.scroll_to_bottom()

    def scroll_to_bottom(self):
        QApplication.processEvents()
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        STATE.stop_event.set()
        MEMORY.save()
        event.accept()
# MAIN
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    threads = [
        threading.Thread(target=TTSManager.worker, daemon=True, name="TTS"),
        threading.Thread(target=InputProcessor.worker, daemon=True, name="Input"),
        threading.Thread(target=MicrophoneManager.worker, daemon=True, name="Mic")
    ]
    for t in threads:
        t.start()

    Logger.log(f"Model: {CONFIG.MODEL_CHAT}")
    Logger.log(f"Memory: {len(MEMORY.data['chat_turns'])} turns loaded")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
