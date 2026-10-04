"""
Helper para usar o Gemini como complemento ao WhisperX.
Função principal: detectar os nomes dos falantes a partir da transcrição.
"""
import json
import os
import re

from google import genai
from google.genai import types


PROMPT_TEMPLATE = """Você é um assistente especializado em análise de transcrições de entrevistas e conversas.

Abaixo está a transcrição de uma conversa entre múltiplos falantes. Cada fala está
rotulada com um identificador técnico no formato [SPEAKER_XX]. Durante a conversa,
os participantes frequentemente dizem seus próprios nomes (ex: "meu nome é João",
"aqui é a Maria", "eu sou o Carlos", "sou a Ana").

Sua tarefa:
1. Identificar o NOME PRÓPRIO de cada SPEAKER_XX.
2. Se um falante não disser o nome dele em nenhum momento, retorne string vazia "".
3. NÃO invente nomes. Se não tiver certeza, retorne "".

Regras importantes:
- Ignore profissões, títulos ("professor", "doutor", "jornalista").
- Nomes próprios geralmente vêm após "meu nome é", "eu sou", "aqui é", "sou o/a".
- Pode haver o mesmo nome sendo mencionado por outros falantes — use o contexto.

Retorne APENAS um JSON válido, sem markdown, sem comentários, no formato exato:
{{"SPEAKER_00": "Nome", "SPEAKER_01": "Nome", "SPEAKER_02": ""}}

Transcrição:
---
{transcricao}
---

Responda APENAS com o JSON:"""

_TITULOS_COMUNS = {
    "dr",
    "dra",
    "prof",
    "professor",
    "professora",
    "engenheiro",
    "engenheira",
    "jornalista",
    "médico",
    "médica",
    "senhor",
    "senhora",
    "sr",
    "sra",
    "diretor",
    "diretora",
    "gerente",
}
_CONTINUACOES = {
    "e",
    "and",
    "y",
    "et",
    "eu",
    "i",
    "yo",
    "je",
    "mas",
    "but",
    "pero",
    "que",
    "that",
    "porque",
    "because",
    "aqui",
    "aí",
    "ai",
    "estou",
    "sou",
    "trabalho",
}
_PADROES_APRESENTACAO = (
    r"\bmeu nome (?:é|e)\s+(?:o\s+|a\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bme chamo\s+(?:o\s+|a\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bpode me chamar de\s+(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\baqui quem fala é\s+(?:o\s+|a\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\baqui é\s+(?:o\s+|a\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\beu sou\s+(?:(?:o|a|sr\.?|sra\.?|dr\.?|dra\.?|professor(?:a)?)\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bsou\s+(?:(?:o|a|sr\.?|sra\.?|dr\.?|dra\.?|professor(?:a)?)\s+)?(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bmy name is\s+(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bI am\s+(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bme llamo\s+(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
    r"\bmi nombre es\s+(?P<nome>[\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[\wÀ-ÖØ-öø-ÿ'’-]+){0,3})",
)


class GeminiRequestError(ValueError):
    """Erro explícito na chamada ao Gemini, com sugestões locais disponíveis."""

    def __init__(self, message: str, sugestoes_locais: dict[str, str]):
        super().__init__(message)
        self.sugestoes_locais = sugestoes_locais


def _limpar_resposta(texto: str) -> str:
    """Remove cercas de markdown que o Gemini às vezes adiciona."""
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)
    return texto.strip()


def _normalizar_chave_api(api_key: str) -> str:
    """Valida se a chave parece uma API key do Google AI Studio."""
    chave = str(api_key or "").strip()
    if not chave:
        raise ValueError("GEMINI_API_KEY não configurada.")

    if re.match(r"^(?:Bearer\s+)?(?:ya29\.|1//)", chave, flags=re.IGNORECASE):
        raise ValueError(
            "A chave informada parece ser um token OAuth/Google Cloud, não uma API key do Google AI Studio. "
            "Use uma 'API key' em https://aistudio.google.com/app/apikey"
        )

    if re.match(r"^Bearer\s+", chave, flags=re.IGNORECASE):
        raise ValueError(
            "Informe apenas a API key do Google AI Studio, sem o prefixo 'Bearer'."
        )

    if re.search(r"\s", chave):
        raise ValueError(
            "A chave do Gemini parece inválida. Copie a API key completa, sem espaços, quebras de linha ou prefixo 'Bearer'."
        )

    return chave


def _limpar_nome(nome: str) -> str:
    nome = re.sub(r"\s+", " ", nome.strip().strip('"\''))
    palavras = nome.split()
    while palavras and palavras[-1].rstrip(".,!?;:").casefold() in _CONTINUACOES:
        palavras.pop()
    nome = " ".join(palavras)
    nome = re.sub(
        r"^(?:" + "|".join(
            re.escape(t) for t in sorted(_TITULOS_COMUNS, key=len, reverse=True)
        ) + r")\.?(?=\s|$)\s*",
        "",
        nome,
        flags=re.IGNORECASE,
    )
    return nome.strip(" .,!?;:")


def _extrair_nomes_heuristicos(transcricao: str) -> dict[str, str]:
    """Sugere nomes próprios a partir de apresentações explícitas dos próprios falantes."""
    evidencias: dict[str, dict[str, tuple[str, int]]] = {}
    blocos = re.finditer(
        r"\[(SPEAKER_\d+)\]\s*(.*?)(?=\n\s*\[SPEAKER_\d+\]|\Z)",
        transcricao or "",
        flags=re.DOTALL,
    )

    for bloco in blocos:
        speaker, fala = bloco.groups()
        for padrao in _PADROES_APRESENTACAO:
            for match in re.finditer(padrao, fala, flags=re.IGNORECASE):
                nome = _limpar_nome(match.group("nome"))
                if not nome or nome.casefold() in _TITULOS_COMUNS:
                    continue
                chave = nome.casefold()
                nomes = evidencias.setdefault(speaker, {})
                exibicao, contagem = nomes.get(chave, (nome, 0))
                nomes[chave] = (exibicao, contagem + 1)

    resultado: dict[str, str] = {}
    for speaker, nomes in evidencias.items():
        ordenados = sorted(nomes.values(), key=lambda item: item[1], reverse=True)
        if len(ordenados) == 1 or ordenados[0][1] > ordenados[1][1]:
            resultado[speaker] = ordenados[0][0]
    return resultado


def detectar_nomes_falantes(
    transcricao: str,
    api_key: str,
    model: str = "gemini-3.8-flash",
    max_chars: int = 12000,
) -> dict[str, str]:
    """
    Envia a transcrição ao Gemini e retorna um dict {SPEAKER_XX: "Nome"}.

    Args:
        transcricao: texto completo com falas no formato [SPEAKER_XX] ...
        api_key: chave da API do Google AI Studio
        model: nome do modelo Gemini
        max_chars: trunca a transcrição para economizar tokens

    Returns:
        dict no formato {"SPEAKER_00": "Michael", "SPEAKER_01": "Ana", ...}
    """
    if not api_key or not api_key.strip():
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY ou GOOGLE_API_KEY não configurada.")

    texto = (transcricao or "")[:max_chars]
    chaves_heuristicas = _extrair_nomes_heuristicos(texto)

    try:
        chave = _normalizar_chave_api(api_key)
    except ValueError as e:
        raise GeminiRequestError(str(e), chaves_heuristicas) from e

    try:
        client = genai.Client(api_key=chave)
        prompt = PROMPT_TEMPLATE.format(transcricao=texto)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        if "UNAUTHENTICATED" in str(e) or "401" in str(e) or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in str(e):
            raise GeminiRequestError(
                "Credenciais de autenticação inválidas para o Gemini. "
                "Use uma API key ativa do Google AI Studio (não um token OAuth). "
                "Gere ou confira a chave em https://aistudio.google.com/app/apikey",
                chaves_heuristicas,
            ) from e
        raise GeminiRequestError(
            f"Erro ao consultar o Gemini: {e}",
            chaves_heuristicas,
        ) from e

    raw = _limpar_resposta(response.text or "")
    if not raw:
        return chaves_heuristicas

    try:
        resultado = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini não retornou JSON válido.\nResposta crua:\n{raw}"
        ) from e

    if not isinstance(resultado, dict):
        raise ValueError("Gemini retornou JSON válido, mas o conteúdo não é um objeto.")

    resultado_limpo = dict(chaves_heuristicas)
    speakers_validos = set(re.findall(r"\[(SPEAKER_\d+)\]", texto))
    for k, v in resultado.items():
        chave_limpa = str(k).replace('"', '').strip()
        if chave_limpa in speakers_validos:
            nome = _limpar_nome(v) if isinstance(v, str) else ""
            resultado_limpo[chave_limpa] = nome or chaves_heuristicas.get(chave_limpa, "")

    if resultado_limpo:
        return resultado_limpo

    return chaves_heuristicas