"""
CS588 PA2 — Agentic AI with ALFWorld + Gemma3-27B
==================================================
Runs a ReAct agent on 15 ALFWorld text episodes using the Gemini API.
Students observe the trajectories and write an analysis report.

Usage:
  export GEMINI_API_KEY="your-key"
  cd alfworld_assignment
  python src/main.py
"""

import os
import re
import sys
import json
import time
import datetime
import google.generativeai as genai
from termcolor import colored
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment import get_environment

# ── Student configuration ───────────────────────────────
STUDENT_ID = "20254163_Minjun_Kim"          # ← Change this
NUM_EPISODES = 15
MAX_STEPS = 30
STEP_DELAY = 1  # seconds between API calls 
# ─────────────────────────────────────────────────────────

# ── Model setup ─────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    sys.exit("Set GEMINI_API_KEY first. Get a free key at https://aistudio.google.com/apikey")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemma-3-27b-it")


# ── Helper functions ────────────────────────────────────

def call_gemini(prompt, max_retries=5):
    """Call Gemini with automatic retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            if any(k in str(e).lower() for k in ("429", "quota", "resourceexhausted", "too many")):
                wait = min(2 ** (attempt + 2), 60)
                print(f"  [rate limit] waiting {wait}s ... (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini API: max retries exceeded")


def parse_receptacles(text):
    return re.findall(r"a ([a-z]+ \d+)", text)


def find_receptacle(valid, *keywords):
    for r in valid:
        for kw in keywords:
            if kw in r:
                return r
    return keywords[0] + " 1"


def detect_task_plan(task_desc, valid_receptacles):
    """Return a step-by-step plan string based on task type."""
    td = task_desc.lower()
    if "cool" in td and "put" in td:
        f = find_receptacle(valid_receptacles, "fridge")
        return (f"PLAN:\n  1. FIND the object\n  2. TAKE it\n  3. go to {f}\n"
                f"  4. cool [obj] with {f}\n  5. go to destination\n  6. put [obj]\n"
                f"  WARNING: Do NOT open fridge — cool handles it.")
    elif "heat" in td and "put" in td:
        m = find_receptacle(valid_receptacles, "microwave")
        return (f"PLAN:\n  1. FIND the object\n  2. TAKE it\n  3. go to {m}\n"
                f"  4. heat [obj] with {m}\n  5. go to destination\n  6. put [obj]\n"
                f"  WARNING: Do NOT open microwave — heat handles it.")
    elif "clean" in td and "put" in td:
        s = find_receptacle(valid_receptacles, "sinkbasin", "sink")
        return (f"PLAN:\n  1. FIND the object\n  2. TAKE it\n  3. go to {s}\n"
                f"  4. clean [obj] with {s}\n  5. go to destination\n  6. put [obj]")
    elif ("look" in td or "examine" in td) and ("lamp" in td or "light" in td):
        furn = [r for r in valid_receptacles if any(k in r for k in ("desk", "sidetable", "dresser"))]
        fs = ", ".join(furn[:3]) if furn else "desk 1, sidetable 1"
        return (f"PLAN:\n  1. FIND the object\n  2. TAKE it\n  3. go to furniture with lamp: {fs}\n"
                f"  4. use desklamp 1 (or floorlamp 1)\n"
                f"  WARNING: Do NOT 'go to desklamp' — lamps are NOT receptacles.")
    elif "put" in td:
        return "PLAN:\n  1. FIND the object\n  2. TAKE it\n  3. go to destination\n  4. put [obj]"
    return "PLAN: Explore the environment and complete the task."


def augment_admissible(admissible, last_obs, curr_recep, inventory):
    """Expand admissible commands based on observation text."""
    aug = [c.replace("in/on", "in") for c in admissible]
    visible = re.findall(r"a ([a-z]+ \d+)", last_obs)
    m = re.match(r"(?:On|In) the (.+?), you see", last_obs)
    if m:
        curr_recep = m.group(1)
    holding = "hands empty" not in inventory.lower()
    if curr_recep:
        if not holding:
            for obj in visible:
                c = f"take {obj} from {curr_recep}"
                if c not in aug:
                    aug.append(c)
        else:
            hm = re.search(r"holding: (.+)", inventory.lower())
            if hm:
                held = hm.group(1).strip()
                for prefix, kw in [("put", ""), ("cool", "fridge"), ("heat", "microwave"), ("clean", "sink")]:
                    if kw and kw not in curr_recep and "basin" not in curr_recep:
                        continue
                    c = f"{prefix} {held} {'in' if prefix == 'put' else 'with'} {curr_recep}"
                    if c not in aug:
                        aug.append(c)
    return aug


def match_to_admissible(action, admissible):
    """Match model output to closest valid command."""
    a = re.sub(r"[*_`]", "", action.strip().split("\n")[0])
    a = re.sub(r"^Action:\s*", "", a, flags=re.IGNORECASE)
    a = a.strip("\"'").replace("in/on", "in")
    a = re.sub(r"\s+", " ", a).strip().lower()
    for cmd in admissible:
        if a == cmd.lower():
            return cmd
    words = set(a.split())
    best, best_n = None, 0
    for cmd in admissible:
        n = len(words & set(cmd.lower().split()))
        if n > best_n:
            best, best_n = cmd, n
    return best if best and best_n >= 2 else a


def track_inventory(action, obs, current):
    al, ol = action.lower(), obs.lower()
    if al.startswith("take ") and "nothing happens" not in ol:
        m = re.match(r"take (.+?) from", al)
        if m:
            return f"You are holding: {m.group(1)}"
    elif al.startswith("put ") and "nothing happens" not in ol:
        return "You are not holding anything (hands empty)."
    elif al == "inventory":
        return obs.strip()
    return current


# ── Environment setup ───────────────────────────────────

def setup_environment(config_path="configs/assignment_config.yaml"):
    orig_argv = sys.argv
    sys.argv = [sys.argv[0], os.path.abspath(config_path)]
    try:
        config = generic.load_config()
    finally:
        sys.argv = orig_argv
    env_type = config["env"]["type"]
    env = get_environment(env_type)(config, train_eval="eval_out_of_distribution")
    return env.init_env(batch_size=1)


# ── Episode runner ──────────────────────────────────────

def run_episode(env, ep_num):
    obs, info = env.reset()
    initial_text = obs[0] if isinstance(obs[0], str) else ""

    if "Your task is to:" in initial_text:
        task_desc = "Your task is to: " + initial_text.split("Your task is to:")[-1].strip()
    else:
        task_desc = "Explore the environment and interact with objects."

    valid_receptacles = parse_receptacles(initial_text)
    admissible = info.get("admissible_commands", [[]])[0]
    plan = detect_task_plan(task_desc, valid_receptacles)

    # ── Episode header ──
    print(colored(f"\n{'='*60}", "cyan"))
    print(colored(f"EPISODE {ep_num}/{NUM_EPISODES}", "yellow", attrs=["bold"]))
    print(colored(f"{'='*60}", "cyan"))
    print(colored(f"Initial Observation: {initial_text.strip()}", "white"))
    print(colored("-" * 60, "cyan"))
    print(colored(f"Task: {task_desc}", "yellow", attrs=["bold"]))
    print(colored(f"Valid receptacles: {', '.join(valid_receptacles)}", "green"))
    print(colored(f"Admissible commands: {len(admissible)}", "green"))
    print(colored(f"Plan:\n{plan}", "magenta"))
    print(colored("=" * 60, "cyan"))

    log = {"student_id": STUDENT_ID, "episode": ep_num, "task": task_desc,
           "success": False, "score": 0, "total_steps": 0, "trajectory": []}

    history = ""
    inventory = "You are not holding anything (hands empty)."
    curr_recep = ""
    last_obs = initial_text
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        full_cmds = augment_admissible(admissible, last_obs, curr_recep, inventory)
        recep_list = ", ".join(valid_receptacles)
        cmd_str = "\n".join(f"  - {c}" for c in full_cmds if c != "help")

        print(colored(f"\n--- Step {step} ---", "yellow", attrs=["bold"]))
        print(colored(f"Inventory: {inventory}", "blue"))
        print(colored(f"Admissible: {full_cmds}", "white"))

        prompt = f"""You are an agent in a text-based household environment.

TASK: {task_desc}

{plan}

ENVIRONMENT: {initial_text}
Valid receptacles: {recep_list}

INVENTORY: {inventory}

HISTORY:
{history}

AVAILABLE ACTIONS:
{cmd_str}

RULES:
1. Follow the plan step by step. Do NOT skip steps.
2. Every object/receptacle MUST include its number (e.g. "desk 1").
3. Only use names from the valid receptacle list.
4. To find objects, "go to" receptacles. "look"/"examine" do NOT reveal new objects.
5. You can hold only 1 item at a time.
6. Open containers before taking or putting items.
7. "Nothing happens" = invalid action. Try a DIFFERENT action.

Reply in EXACTLY this format:
Thought: [which plan step you are on and what to do next]
Action: [one action from the available list]"""

        try:
            text = call_gemini(prompt)
            if "Action:" in text:
                thought = text.split("Action:")[0].replace("Thought:", "").strip()
                action = text.split("Action:")[-1].strip().split("\n")[0].strip()
            else:
                thought = "Could not parse thought."
                action = text.split("\n")[-1].strip()
            action = match_to_admissible(action, full_cmds)
        except Exception as e:
            thought, action = f"Error: {e}", "look"

        print(colored(f"Thought: {thought}", "magenta"))
        print(colored(f"Action:  {action}", "green", attrs=["bold"]))

        obs, _, dones, infos = env.step([action])
        text_obs = obs[0]

        print(colored(f"Observation: {text_obs}", "cyan"))

        # Debug info on invalid actions
        if text_obs.strip() == "Nothing happens.":
            print(colored(f"  [DEBUG] Sent action repr: {repr(action)}", "red"))
            print(colored(f"  [DEBUG] Admissible: {admissible[:8]}...", "red"))

        admissible = infos.get("admissible_commands", [[]])[0]
        last_obs = text_obs
        m = re.match(r"(?:On|In) the (.+?), you see", text_obs)
        if m:
            curr_recep = m.group(1)
        inventory = track_inventory(action, text_obs, inventory)

        log["trajectory"].append({"step": step, "thought": thought,
                                  "action": action, "observation": text_obs})

        history += f"> {action}\n{text_obs}\n"
        if len(history.split("\n")) > 40:
            history = "\n".join(history.split("\n")[-20:])

        done = dones[0]
        if not done:
            time.sleep(STEP_DELAY)

    is_success = infos.get("won", [False])[0]
    log["success"] = bool(is_success)
    log["score"] = 1 if is_success else 0
    log["total_steps"] = len(log["trajectory"])

    print(colored(f"\n{'='*60}", "cyan"))
    if is_success:
        print(colored(f"Result: SUCCESS in {log['total_steps']} steps", "green", attrs=["bold"]))
    else:
        print(colored(f"Result: FAILED after {log['total_steps']} steps", "red", attrs=["bold"]))

    return log


# ── Main ────────────────────────────────────────────────

def main():
    env = setup_environment()
    results = []

    for ep in range(1, NUM_EPISODES + 1):
        try:
            log = run_episode(env, ep)
        except Exception as e:
            print(f"  >> ERROR ep {ep}: {e}")
            log = {"student_id": STUDENT_ID, "episode": ep, "task": "ERROR",
                   "success": False, "score": 0, "total_steps": 0,
                   "error": str(e), "trajectory": []}
        results.append(log)

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"eval_results/react_{STUDENT_ID}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {path}")

    # Summary
    successes = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"RESULTS: {successes}/{NUM_EPISODES} succeeded ({100*successes/NUM_EPISODES:.0f}%)")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        tag = "OK" if r["success"] else "FAIL"
        print(f"  ep{i+1:02d} [{tag:4s}] {r['total_steps']:2d} steps | {r['task'][:50]}")


if __name__ == "__main__":
    main()
