# 1. We will load a language model model from huggingface (Qwen 0.5B Instruct)
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"    # swap if you prefer another instruct model
LOAD_8BIT    =  True                       # set True if you installed bitsandbytes and want 8-bit loading
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ====== TODO ======
# Load model with AutoModelForCausalLM.from_pretrained() from huggingface with the above MODEL_NAME, LOAD_8BIT, DTYPE
# Try to load the model; fall back to None with a friendly warning if loading fails
model = None
try:
    if LOAD_8BIT:
        # bitsandbytes 8-bit loading (requires bitsandbytes installed)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
        # move to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
except Exception as e:
    # Leave model as None and print a warning (not raising)
    print(f"Warning: failed to load model '{MODEL_NAME}': {e}")

# Generation configuration: use GenerationConfig to define the generation parameters
gen_cfg = GenerationConfig(
    max_new_tokens=128,
    temperature=0.0,
    top_p=0.9,
    do_sample=False,
)
# ====== TODO ======

# ====== Helper function: Enforce two-line schema in the decoding ======
T_PATTERN = re.compile(r"Thought:\s*(.+)")
A_PATTERN = re.compile(r"Action:\s*(.+)")

def _postprocess_to_two_lines(text: str) -> str:
    """
    Extract the first 'Thought:' and 'Action:' lines from the model output.
    If the model drifts, fall back to a conservative default Action.
    """
    # Stop at first Observation if present (model shouldn't produce it, but just in case)
    text = text.split("\nObservation:")[0]
    # Keep only the assistant's new tokens (strip any trailing commentary)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Try to find explicit Thought/Action anywhere in the output
    thought = None
    action  = None
    for ln in lines:
        if thought is None:
            m = T_PATTERN.match(ln)
            if m:
                thought = m.group(1).strip()
                continue
        if action is None:
            m = A_PATTERN.match(ln)
            if m:
                action = m.group(1).strip()
                continue

    # Fallbacks if the model didn’t comply perfectly
    if thought is None:
        thought = "I should search for key facts related to the question."
    if action is None:
        # Default to a generic search; your controller will parse it.
        action = 'search[query="(auto) refine the user question", k=3]'

    return f"Thought: {thought}\nAction: {action}"
# ====== Helper function: Enforce two-line schema in the decoding ======



# 2. We define the LLM function. This will be plugged into the agent without changing the controller ---
def hf_llm(prompt: str) -> str:
    """
    Completes from your existing ReAct prompt and returns exactly two lines:
    'Thought: ...' and 'Action: ...'
    """
    # We add a strong instruction to the prompt to improve compliance with the format
    format_guard = (
        "\n\nIMPORTANT: Respond with EXACTLY two lines in this format:\n"
        "Thought: <one concise sentence>\n"
        "Action: <either search[query=\"...\"] or finish[answer=\"...\"]>\n"
        "Do NOT include Observation."
    )
    full_prompt = prompt + format_guard

    # ====== TODO ======
    #     Here, let's write the code to use language model to generate the response given the full_prompt
    #     First, we need to use the tokenizer to tokenize the prompt into pytorch tensors
    #     Second, we need to use model.generate() to generate the model response (which includes the Thought and Action)
    # If model failed to load, return a conservative fallback action
    if model is None:
        # Attempt to extract a concise query from the prompt for the fallback
        q = prompt.split("User Question:")[-1].split("\n")[0].strip() if "User Question:" in prompt else "(the user question)"
        fallback = f"Thought: I should search for key facts related to the question.\nAction: search[query=\"{q}\", k=3]"
        return _postprocess_to_two_lines(fallback)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        try:
            output_ids = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=gen_cfg.max_new_tokens, do_sample=gen_cfg.do_sample, pad_token_id=tokenizer.eos_token_id)
        except TypeError:
            # Older transformers may not accept generation_config; pass directly
            output_ids = model.generate(**inputs, max_new_tokens=gen_cfg.max_new_tokens, do_sample=gen_cfg.do_sample, pad_token_id=tokenizer.eos_token_id)
    # ====== TODO ======


    # Slice off the prompt tokens to get only the completion
    start_idx = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, start_idx:]
    completion = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)

    return _postprocess_to_two_lines(completion)

# We will wire it into the agent system
LLM = hf_llm