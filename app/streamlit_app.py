"""Interface de démonstration : chat client + bouton de confirmation des tickets + monitoring.

Lancement (l'API doit tourner) :  streamlit run app/streamlit_app.py
"""
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CUSTOMERS = {"C001": "Salma Bennani", "C002": "Youssef El Idrissi", "C003": "Imane Tazi"}

st.set_page_config(page_title="ShopNova — Support", page_icon="🛍️", layout="wide")


def api(method: str, path: str, customer_id: str | None = None, **kwargs):
    headers = {"X-Customer-Id": customer_id} if customer_id else {}
    try:
        response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Erreur de l'API : {exc}")
        return None


# ------------------------------------------------------------------ barre latérale : session simulée
with st.sidebar:
    st.header("Session (simulée)")
    customer_id = st.selectbox("Client connecté", list(CUSTOMERS), format_func=lambda c: f"{c} — {CUSTOMERS[c]}")
    if st.session_state.get("customer_id") != customer_id:   # changement de client : nouvelle conversation
        st.session_state.customer_id = customer_id
        st.session_state.messages = []
        st.session_state.draft = None
    if st.button("Nouvelle conversation"):
        st.session_state.messages, st.session_state.draft = [], None
    health = api("GET", "/health")
    if health:
        st.caption(f"API : {health['status']} · modèle {health['model']} · prompt {health['prompt_version']} · "
                   f"version {health['agent_version']}")

tab_chat, tab_monitoring = st.tabs(["💬 Assistant", "📊 Monitoring"])

# ------------------------------------------------------------------ onglet chat
with tab_chat:
    st.title("Assistant du service client ShopNova")
    st.caption("Commandes, retours, livraison, garantie, paiement. Exemples de commandes : CMD-10001, CMD-10002.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("details"):
                st.caption(message["details"])

    draft = st.session_state.draft
    if draft:  # supervision humaine : le ticket n'existe qu'après ce clic
        st.warning(f"📝 Demande d'escalade préparée (catégorie : **{draft['category']}**). "
                   "Voulez-vous la transmettre à un conseiller ?")
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Confirmer le ticket"):
            ticket = api("POST", f"/tickets/{draft['id']}/confirm", customer_id)
            if ticket:
                text = f"Votre ticket **{ticket['ticket_id']}** est créé. Un conseiller vous recontactera."
                st.session_state.messages.append({"role": "assistant", "content": text})
            st.session_state.draft = None
            st.rerun()
        if col_no.button("❌ Annuler"):
            api("POST", f"/tickets/{draft['id']}/cancel", customer_id)
            st.session_state.messages.append({"role": "assistant", "content": "Demande annulée."})
            st.session_state.draft = None
            st.rerun()

    if question := st.chat_input("Votre question..."):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-10:]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("L'assistant réfléchit..."):
            r = api("POST", "/chat", customer_id, json={"message": question, "history": history})
        if r:
            details = f"outils : {', '.join(r['tools_used']) or 'aucun'} · {r['latency_ms']:.0f} ms · {r['outcome']}"
            if r["sources"]:
                details += f" · sources : {' ; '.join(r['sources'][:3])}"
            st.session_state.messages.append({"role": "assistant", "content": r["answer"], "details": details})
            if r["draft_id"]:
                st.session_state.draft = {"id": r["draft_id"], "category": r["draft_category"]}
        st.rerun()

# ------------------------------------------------------------------ onglet monitoring
with tab_monitoring:
    st.title("Monitoring de l'agent")
    if st.button("🔄 Actualiser"):
        st.rerun()
    m = api("GET", "/metrics")
    if m and m.get("n_requests"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requêtes", m["n_requests"])
        c2.metric("Taux d'erreur", f"{m['error_rate']:.0%}")
        c3.metric("Latence P95", f"{m['latency_p95_ms'] / 1000:.1f} s")
        c4.metric("Coût total", f"{m['cost_total_usd']:.4f} $")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Bloquées (garde-fous)", f"{m['blocked_rate']:.0%}")
        c6.metric("Escalades proposées", f"{m['escalation_rate']:.0%}")
        c7.metric("Tickets confirmés", m["tickets_confirmed"])
        c8.metric("Erreurs d'outils", f"{m['tool_error_rate']:.0%}")
        left, right = st.columns(2)
        left.subheader("Utilisation des outils")
        left.bar_chart(m["tool_usage"])
        right.subheader("Issues des requêtes")
        right.bar_chart(m["outcomes"])
        with st.expander("Toutes les métriques (JSON)"):
            st.json(m)
    elif m is not None:
        st.info("Aucune trace pour l'instant : pose quelques questions dans l'onglet Assistant.")