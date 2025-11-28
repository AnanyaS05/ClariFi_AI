# Step 4: Integrating Components into an Agent System

# Finally, we will put everything together as an agent system, including the external database, the information search tools, and a base language model.

# The agent will take the input and use a language model to output the Thought and Action.
# The agent will execute the Action (which in this case is searching for a document) and concatenate the returned document as Observation to the next round of the prompt
# The agent will iterate over the above two steps until it identifies the answer or reaches an iteration limit.

# ----------------------------
# We will define the agent controller that combines everything we define above
# ----------------------------
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import json, math, re, textwrap, random, os, sys
import math
import inspect
from collections import Counter, defaultdict

# import python files from the same folder, such as language_model.py, knowledge_base.py, prompting_techniques.py
# Add the current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from prompting_techniques import make_prompt, parse_action

@dataclass
class Step:
    thought: str
    action: str
    observation: str

@dataclass
class AgentConfig:
    max_steps: int = 6
    # allow both search and explain_term tools
    allow_tools: Tuple[str, ...] = ("search", "explain_term", "list_docs")
    verbose: bool = True

class ReActAgent:
    def __init__(
        self,
        llm: Callable[[str], str],
        tools: Dict[str, Dict[str, Any]],
        config: Optional[AgentConfig] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []

    # ----- internal helper -----
    @staticmethod
    def _extract_last_thought_action(text: str) -> Tuple[str, str]:
        """
        Split the model output into:
          - thought: everything BEFORE the last 'Action:'
          - action_line: the last 'Action:' line (including its arguments)

        This does NOT require the model to literally write 'Thought:'.
        """
        if not isinstance(text, str):
            return "(no thought)", 'Action: finish[answer="(no action)"]'

        # Find the last occurrence of 'Action:'
        idx = text.rfind("Action:")
        if idx == -1:
            # No action at all – treat the whole output as the thought
            return text.strip(), 'Action: finish[answer="(no action)"]'

        thought_part = text[:idx].strip()
        action_part = text[idx:].strip()  # starts with 'Action:'

        # If, for some reason, there are multiple 'Action:' tokens inside,
        # just keep the first one in this tail.
        # (parse_action will handle it as long as format is Action: name[...])
        first_newline = action_part.find("\n")
        if first_newline != -1:
            # Take only the first line that starts with "Action:"
            action_line = action_part[:first_newline]
        else:
            action_line = action_part

        return thought_part or "(no thought)", action_line
    @staticmethod
    def _clean_thought(thought_text: str) -> str:
        """
        Remove a leading 'Thought:' prefix if present, and strip whitespace.
        """
        if not isinstance(thought_text, str):
            return "(no thought)"
        t = thought_text.strip()
        # If the model included the 'Thought:' label, strip it
        if t.lower().startswith("thought:"):
            t = t.split(":", 1)[1].strip()
        return t or "(no thought)"


    # ----- main loop -----
    def run(self, user_query: str) -> Dict[str, Any]:
        self.trajectory.clear()
        final_answer: Optional[str] = None  # filled when we see finish[...]

        for step_idx in range(self.config.max_steps):
            # 1. Build prompt from user query + trajectory
            traj_dicts = [asdict(s) for s in self.trajectory]
            prompt = make_prompt(user_query, traj_dicts)

            if self.config.verbose:
                print(f"\n--- Step {step_idx + 1} Prompt (truncated) ---")
                for line in prompt.splitlines()[-20:]:
                    print(line)

            # 2. Call the language model
            try:
                out = self.llm(prompt)
            except Exception as e:
                out = (
                    "I encountered an error while generating a response.\n"
                    f"Action: finish[answer=\"Tool error calling LLM: {e}\"]"
                )

            if self.config.verbose:
                print("\n--- LLM raw output ---")
                print(out)

            # 3. Extract thought and action from raw output
            raw_thought, action_line = self._extract_last_thought_action(out)
            # 🔹 CLEAN the thought so it does NOT include 'Thought: '
            thought = self._clean_thought(raw_thought)

            # 4. Parse the action line
            parsed = parse_action(action_line)
            if not parsed:
                observation = f"Invalid action format: {action_line}"
                # 🔹 Store clean thought + full action_line
                self.trajectory.append(Step(thought, action_line, observation))
                break

            name, args = parsed

            # 5. Handle finish action and capture the answer RIGHT HERE
            if name == "finish":
                ans = args.get("answer", "")
                if not isinstance(ans, str):
                    ans = str(ans)
                final_answer = ans.strip()
                observation = "done"
                # 🔹 Store final step
                self.trajectory.append(Step(thought, action_line, observation))
                break

            # 6. Check tool permissions/existence
            if name not in self.config.allow_tools or name not in self.tools:
                observation = f"Action '{name}' not allowed or not found."
                self.trajectory.append(Step(thought, action_line, observation))
                break

            # 7. Execute the tool
            try:
                tool_fn = self.tools[name]["fn"]
                sig = inspect.signature(tool_fn)
                valid_args = {k: v for k, v in args.items() if k in sig.parameters}
                obs_payload = tool_fn(**valid_args)
                observation = json.dumps(obs_payload, ensure_ascii=False)
            except Exception as e:
                observation = f"Tool error: {e}"

            # 🔹 Store clean thought + full action_line each step
            self.trajectory.append(Step(thought, action_line, observation))

        # ----- Build final answer -----
        if not final_answer:
            # If we NEVER hit finish[answer="..."], fall back to last thought
            if self.trajectory:
                final_answer = self.trajectory[-1].thought
            else:
                final_answer = "I’m sorry, I couldn’t generate an answer."

        return {
            "question": user_query,
            "final_answer": final_answer,
            "steps": [asdict(s) for s in self.trajectory],
        }





