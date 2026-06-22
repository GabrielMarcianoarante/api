import os
import time
import random
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from google import genai

# ===== CONFIGURAÇÃO =====
DOTENV_FILE = Path(__file__).resolve().parent / ".env"


def carregar_env():
    """Carrega variáveis de um .env simples (sem dependência extra)."""
    if DOTENV_FILE.is_file():
        for linha in DOTENV_FILE.read_text(encoding="utf-8").splitlines():
            texto = linha.strip()
            if not texto or texto.startswith("#") or "=" not in texto:
                continue
            chave, valor = texto.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            if chave and valor:
                os.environ.setdefault(chave, valor)


carregar_env()  # só tem efeito localmente; na Vercel as env vars vêm do dashboard

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLIENT_API_KEY = os.getenv("CLIENT_API_KEY", "")  # chave que SEUS clientes vão usar

if not GEMINI_API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY no .env (chave da API do Gemini).")
if not CLIENT_API_KEY:
    raise RuntimeError(
        "Defina CLIENT_API_KEY no .env (chave que você vai distribuir pros seus "
        "próprios PCs clientes, pra ninguém mais usar sua API à toa)."
    )

MAX_RETRIES = 3
INITIAL_WAIT_TIME = 1
MAX_WAIT_TIME = 8

client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI(title="Gemini Relay API")


class PerguntaRequest(BaseModel):
    pergunta: str


class RespostaResponse(BaseModel):
    resposta: str
    tokens_estimados: int | None = None


def verificar_api_key(x_api_key: str | None):
    if not x_api_key or x_api_key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida ou ausente.")


@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(body: PerguntaRequest, x_api_key: str | None = Header(default=None)):
    verificar_api_key(x_api_key)

    pergunta = body.pergunta.strip()
    if not pergunta:
        raise HTTPException(status_code=400, detail="Campo 'pergunta' vazio.")

    prompt = f"Resolva a questão como se fosse um inciante mas com certeza nao precisa explicar bote algums comentarios se quiser nao muitos mas comentarios estilo humano no codigo e se a questao pedir restrição tipo não pode usar def voce vai seguir a restrição ao todo custo: {pergunta}"

    tokens_estimados = None
    try:
        token_count = client.models.count_tokens(
            model="gemini-2.5-flash", contents=prompt
        )
        tokens_estimados = token_count.total_tokens
    except Exception:
        pass  # não trava a requisição só por isso

    resposta_texto = ""
    ultimo_erro = None
    for tentativa in range(MAX_RETRIES):
        try:
            resposta = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            resposta_texto = resposta.text.strip()
            ultimo_erro = None
            break
        except Exception as e:
            ultimo_erro = e
            if tentativa < MAX_RETRIES - 1:
                wait_time = min(INITIAL_WAIT_TIME * (2**tentativa), MAX_WAIT_TIME)
                wait_time += random.uniform(0, 1)
                time.sleep(wait_time)

    if ultimo_erro is not None:
        raise HTTPException(
            status_code=502, detail=f"Erro ao chamar o Gemini: {ultimo_erro}"
        )

    return RespostaResponse(resposta=resposta_texto, tokens_estimados=tokens_estimados)


@app.get("/health")
def health():
    return {"status": "ok"}
