_base_ = "./base.py"

# General Config
tag = "gsm8k_openai"
concurrency = 1
workdir = "workdir"
log_path = "log.txt"
save_path = "gsm8k_openai.jsonl"
use_local_proxy = False  # True for local proxy, False for public proxy

use_hierarchical_agent = False

dataset = dict(
    type="gsm8k_dataset",
    path="openai/gsm8k",
    name="main",
    split="test",
)

general_agent_config = dict(
    type="general_agent",
    name="general_agent",
    model_id="qwen-plus",
    description="A general agent for GSM8K math word problems.",
    max_steps=12,
    template_path="src/agent/general_agent/prompts/general_agent.yaml",
    provide_run_summary=True,
    tools=["python_interpreter_tool"],
)

agent_config = general_agent_config
