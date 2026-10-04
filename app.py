"""
Interface Streamlit para transcrição + diarização com WhisperX.
Com sessão persistente, renomeação de falantes e sugestão de nomes via Gemini.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from transcriber import transcribe_and_diarize
from gemini_helper import GeminiRequestError, detectar_nomes_falantes
from transcript_formatter import format_transcript_by_speaker, transcript_to_docx

load_dotenv()

st.set_page_config(
    page_title="Transcrição com WhisperX + Gemini",
    page_icon="🎙️",
    layout="wide",
)

# ---------- Inicializa estado persistente ----------
DEFAULTS = {
    "result": None,
    "audio_name": None,
    "speaker_map": {},
    "gemini_sugestoes": {},   # sugestões do Gemini separadas
    "gemini_raw_response": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_session():
    """Limpa tudo para começar uma nova transcrição."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = {} if isinstance(v, dict) else None


def get_hf_token() -> str:
    """Load the Hugging Face token from the environment or Streamlit secrets."""
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token

    try:
        token = st.secrets.get("HF_TOKEN", "")
    except (FileNotFoundError, KeyError):
        return ""
    return str(token).strip()


def nome_exibido(spk_id: str) -> str:
    """Retorna o nome customizado ou o ID original."""
    nome = st.session_state.speaker_map.get(spk_id, "").strip()
    return nome if nome else spk_id


def aplicar_sugestoes(sugestoes: dict[str, str]):
    """Guarda sugestões e preenche apenas nomes ainda não editados."""
    st.session_state.gemini_sugestoes = sugestoes
    for spk, nome in sugestoes.items():
        if nome and not st.session_state.speaker_map.get(spk, "").strip():
            st.session_state.speaker_map[spk] = nome


st.title("🎙️ Transcrição de Áudio com Separação de Falantes")
st.caption("WhisperX (transcrição local) + Gemini (detecção de nomes)")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configurações")

    hf_token = get_hf_token()

    gemini_api_key = st.text_input(
        "Chave da API do Gemini",
        value=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
        type="password",
    )

    gemini_model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"

    st.divider()
    st.subheader("🌐 Idioma")

    language = st.selectbox(
        "Idioma do áudio",
        options=["auto", "pt", "en", "es", "fr", "de", "it"],
        index=0,
    )

    st.divider()

    if st.session_state.result is not None:
        if st.button("🗑️ Limpar sessão", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()


# ---------- FLUXO 1: Sem resultado → uploader ----------
if st.session_state.result is None:
    uploaded = st.file_uploader(
        "📂 Carregue um arquivo de áudio",
        type=["mp3", "mp4", "wav", "m4a", "flac", "ogg", "opus", "webm"],
    )

    if uploaded is not None:
        st.audio(uploaded)

        if st.button("🚀 Transcrever", type="primary"):
            if not hf_token:
                st.error(
                    "Configure HF_TOKEN no arquivo .env ou nas secrets do ambiente "
                    "para habilitar a diarização."
                )
                st.stop()

            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            status_box = st.status("Iniciando...", expanded=True)

            def progress(msg: str):
                status_box.update(label=msg, state="running")
                status_box.write(msg)

            try:
                result = transcribe_and_diarize(
                    audio_path=tmp_path,
                    hf_token=hf_token,
                    language=None if language == "auto" else language,
                    progress_callback=progress,
                )
                status_box.update(label="✅ Concluído!", state="complete", expanded=False)

                st.session_state.result = result
                st.session_state.audio_name = uploaded.name
                st.session_state.speaker_map = {spk: "" for spk in result["speakers"]}
                st.session_state.gemini_sugestoes = {}

                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Erro", state="error")
                st.exception(e)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ---------- FLUXO 2: Com resultado ----------
else:
    result = st.session_state.result

    st.success(
        f"Transcrição de **{st.session_state.audio_name}** concluída. "
        f"Idioma: **{result['language']}** — "
        f"Falantes: **{len(result['speakers'])}**"
    )

    # ---------- BOTÃO GEMINI ----------
    st.subheader("🤖 Detecção de nomes com Gemini")
    st.caption(
        "O Gemini analisa a transcrição e cruza o contexto. Se a API estiver indisponível, "
        "usamos apresentações explícitas como 'meu nome é João' ou 'me chamo Ana'. "
        "Confira e corrija as sugestões antes de usar."
    )

    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        if st.button("✨ Sugerir nomes via Gemini", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("Configure a chave da API do Gemini na barra lateral.")
            else:
                with st.spinner(f"Consultando {gemini_model}..."):
                    try:
                        sugestoes = detectar_nomes_falantes(
                            transcricao=result["full_text"],
                            api_key=gemini_api_key,
                            model=gemini_model,
                        )
                        aplicar_sugestoes(sugestoes)

                        st.success(
                            f"Gemini sugeriu {sum(1 for v in sugestoes.values() if v)} nome(s). "
                            "Confira abaixo e ajuste se necessário."
                        )
                        st.rerun()
                    except GeminiRequestError as e:
                        if e.sugestoes_locais:
                            aplicar_sugestoes(e.sugestoes_locais)
                            st.warning(
                                f"{e} O Gemini não forneceu as sugestões; "
                                "os nomes abaixo foram extraídos localmente de apresentações explícitas. "
                                "Revise-os antes de usar."
                            )
                            st.rerun()
                        else:
                            st.error(f"Erro ao consultar Gemini: {e}")
                    except Exception as e:
                        st.error(f"Erro ao consultar Gemini: {e}")

    with col_g2:
        if st.session_state.gemini_sugestoes:
            with st.expander("📋 Resposta do Gemini", expanded=True):
                st.json(st.session_state.gemini_sugestoes)

    # ---------- Mapeamento de falantes ----------
    with st.expander("✏️ Nomear / corrigir falantes", expanded=True):
        st.caption(
            "Os nomes aparecerão em todas as abas e nos downloads. "
            "Deixe vazio para manter o ID original (SPEAKER_XX)."
        )

        # Prévia do primeiro trecho de cada falante
        primeiros_trechos = {}
        for spk in result["speakers"]:
            trechos = [s["text"] for s in result["segments"] if s["speaker"] == spk][:1]
            primeiros_trechos[spk] = trechos[0] if trechos else ""

        cols = st.columns(min(3, len(result["speakers"])) or 1)
        for i, spk in enumerate(result["speakers"]):
            with cols[i % len(cols)]:
                sugestao_gemini = st.session_state.gemini_sugestoes.get(spk, "")
                badge = " 🤖" if sugestao_gemini else ""

                novo_valor = st.text_input(
                    f"🏷️ {spk}{badge}",
                    value=st.session_state.speaker_map.get(spk, ""),
                    key=f"input_{spk}",
                )
                st.session_state.speaker_map[spk] = novo_valor

                trecho = primeiros_trechos[spk]
                if trecho:
                    st.caption(f"💬 _{trecho[:120]}{'...' if len(trecho) > 120 else ''}_")

    # ---------- Texto final ----------
    full_text_mapped = format_transcript_by_speaker(
        result["full_text"], st.session_state.speaker_map
    )

    tab1, tab2, tab3 = st.tabs(["📋 Por falante", "🕐 Com timestamps", "📄 Texto puro"])

    with tab1:
        st.markdown(full_text_mapped)

    with tab2:
        for seg in result["segments"]:
            st.markdown(
                f"**`{seg['start_hms']} → {seg['end_hms']}`** "
                f"`{nome_exibido(seg['speaker'])}` — {seg['text']}"
            )

    with tab3:
        plain = "\n".join(
            f"{nome_exibido(s['speaker'])}: {s['text']}"
            for s in result["segments"]
        )
        st.text_area("Texto puro", plain, height=400)

    # ---------- Downloads ----------
    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.download_button(
            "⬇️ Baixar por falante (.docx)",
            data=transcript_to_docx(full_text_mapped),
            file_name=f"{Path(st.session_state.audio_name).stem}_transcricao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    with col_b:
        timestamps_text = "\n".join(
            f"[{s['start_hms']} → {s['end_hms']}] "
            f"{nome_exibido(s['speaker'])}: {s['text']}"
            for s in result["segments"]
        )
        st.download_button(
            "⬇️ Baixar com timestamps (.txt)",
            data=timestamps_text,
            file_name=f"{Path(st.session_state.audio_name).stem}_timestamps.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_c:
        plain_text = "\n".join(
            f"{nome_exibido(s['speaker'])}: {s['text']}"
            for s in result["segments"]
        )
        st.download_button(
            "⬇️ Baixar por Texto Puro (.txt)",
            data=plain_text,
            file_name=f"{Path(st.session_state.audio_name).stem}_texto_puro.txt",
            mime="text/plain",
            use_container_width=True,
        )