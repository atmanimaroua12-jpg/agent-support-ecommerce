"""Discussion avec l'agent dans le terminal.  Usage : python scripts/chat.py [C001|C002|C003]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_support.agent import build_agent  # noqa: E402
from agent_support.config import load_config  # noqa: E402

customer_id = sys.argv[1] if len(sys.argv) > 1 else "C001"
agent = build_agent(load_config())
history = []
print(f"Connecté en tant que {customer_id}. Tape 'quit' pour sortir.\n")

while True:
    question = input("Vous : ").strip()
    if question.lower() in {"quit", "exit", "q"}:
        break
    r = agent.run(question, customer_id, history)
    for call in r.tool_calls:
        print(f"   🔧 {call.name}({call.args}) -> {call.status} {call.error_code or ''} [{call.duration_ms} ms]")
    print(f"Agent : {r.answer}")
    if r.error:
        print(f"   ⚠️ Erreur LLM : {r.error}")
    if r.draft:
        print(f"   📝 Brouillon {r.draft['draft_id']} en attente de confirmation (bouton de l'interface, étape 7)")
    print(f"   ⏱ {r.latency_ms:.0f} ms | {r.iterations} tour(s) | {r.input_tokens}+{r.output_tokens} tokens"
          f" | issue : {r.outcome}\n")
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": r.answer}]