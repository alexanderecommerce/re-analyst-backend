from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RE Analyst Proto", version="0.2.0")

# --- CORS ---
FRONTEND_ORIGINS = [
    "http://localhost:3000",                 # lokal beim Entwickeln
    "https://re-analyst-ggwwvihu9-adrians-projects-6115aa33.vercel.app", # deine Vercel-URL nach Deploy
    # "https://deine-domain.de",             # optional eigene Domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#   Requests / Responses
# =========================

class YieldRequest(BaseModel):
    price_eur: float
    monthly_rent_eur: float

class NetYieldRequest(BaseModel):
    price_eur: float
    monthly_rent_eur: float
    non_recoverable_costs_rate: float = 0.20
    vacancy_rate: float = 0.05
    acquisition_costs_rate: float = 0.08

class ScenarioRequest(BaseModel):
    price_eur: float
    monthly_rent_eur: float
    equity_ratio: float = 0.3
    interest_now: float = 0.03
    scenario_rate_delta: float = 0.01
    amort_rate: float = 0.02
    non_recoverable_costs_rate: float = 0.20
    vacancy_rate: float = 0.05

class AnalyzeRequest(BaseModel):
    price_eur: float
    monthly_rent_eur: float
    equity_ratio: float = 0.3
    interest_now: float = 0.03
    scenario_rate_delta: float = 0.01
    amort_rate: float = 0.02
    non_recoverable_costs_rate: float = 0.20
    vacancy_rate: float = 0.05
    acquisition_costs_rate: float = 0.08

class ChatRequest(BaseModel):
    message: str
    context: dict | None = None

class ChatResponse(BaseModel):
    reply: str
    context_out: dict | None = None

# =========================
#   Utility functions
# =========================

def _to_float(s: str) -> float | None:
    s = s.strip().lower().replace("€", "").replace(" ", "")
    if s.endswith("k"):
        return float(s[:-1].replace(",", ".")) * 1000
    if "tsd" in s:
        return float(s.replace("tsd", "").replace(",", ".")) * 1000
    if "," in s and s.count(",") == 1 and (s.rsplit(",", 1)[-1].isdigit()):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None

def _extract_params(text: str) -> dict:
    t = text.lower()
    params: dict = {}

    # Preis
    m = re.search(r"(kaufpreis|preis|purchase|price)\D*([0-9\.\,ktsd\s€]+)", t)
    if m: params["price_eur"] = _to_float(m.group(2))

    # Miete
    m = re.search(r"(miete|kaltmiete|rent)\D*([0-9\.\,ktsd\s€]+)", t)
    if m: params["monthly_rent_eur"] = _to_float(m.group(2))

    # EK-Quote
    m = re.search(r"(ek|ek-quote|eigenkapital|equity)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["equity_ratio"] = v/100 if v > 1 else v

    # Zins
    m = re.search(r"(zins|interest)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["interest_now"] = v/100 if v > 1 else v

    # Delta
    m = re.search(r"(delta|szenario|steigt|erhöht|\+)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["scenario_rate_delta"] = v/100 if v > 1 else v

    # Leerstand
    m = re.search(r"(leerstand|vacancy)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["vacancy_rate"] = v/100 if v > 1 else v

    # NK-Quote
    m = re.search(r"(nicht\s*umlagef(ä|ae)hig|bewirtschaftung|operating\s*costs)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(3))
        if v is not None: params["non_recoverable_costs_rate"] = v/100 if v > 1 else v

    # Tilgung
    m = re.search(r"(tilgung|amortisation|amort)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["amort_rate"] = v/100 if v > 1 else v

    # Kaufnebenkosten
    m = re.search(r"(kaufnebenkosten|grunderwerb|notar|makler|acquisition\s*costs)\D*([0-9\.\,]+)\s*%?", t)
    if m:
        v = _to_float(m.group(2))
        if v is not None: params["acquisition_costs_rate"] = v/100 if v > 1 else v

    # Anpassungen
    m = re.search(r"(miete)\s*(auf|=)\s*([0-9\.\,ktsd\s€]+)", t)
    if m: params["monthly_rent_eur"] = _to_float(m.group(3))
    m = re.search(r"(preis|kaufpreis)\s*(auf|=)\s*([0-9\.\,ktsd\s€]+)", t)
    if m: params["price_eur"] = _to_float(m.group(3))

    return {k: v for k, v in params.items() if v is not None}

def _analyze(price_eur, monthly_rent_eur,
             equity_ratio=0.3, interest_now=0.03, scenario_rate_delta=0.01,
             amort_rate=0.02, non_recoverable_costs_rate=0.20,
             vacancy_rate=0.05, acquisition_costs_rate=0.08):
    gross_yield = (monthly_rent_eur * 12) / price_eur * 100
    net_monthly = monthly_rent_eur * (1 - non_recoverable_costs_rate) * (1 - vacancy_rate)
    annual_net = net_monthly * 12
    total_invest = price_eur * (1 + acquisition_costs_rate)
    net_yield = (annual_net / total_invest) * 100
    noi = annual_net
    loan = price_eur * (1 - equity_ratio)
    new_rate = interest_now + scenario_rate_delta
    annual_interest = loan * new_rate
    annual_amort = loan * amort_rate
    annual_debt_service = annual_interest + annual_amort
    dscr = noi / annual_debt_service if annual_debt_service > 0 else None
    return {
        "gross_yield_pct": round(gross_yield, 2),
        "net_yield_pct": round(net_yield, 2),
        "scenario": {
            "new_rate": round(new_rate, 4),
            "annual_debt_service_eur": round(annual_debt_service, 2),
            "dscr": round(dscr, 2) if dscr else None,
        },
    }

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

def _make_commentary(metrics: dict, assumptions: dict) -> str:
    if not USE_OPENAI:
        return f"Kommentar (ohne KI): Brutto {metrics['gross_yield_pct']}%, Netto {metrics['net_yield_pct']}%, DSCR {metrics['scenario']['dscr']}."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        system = "Du bist ein vorsichtiger Immobilien-Analyst (DACH). Kurz, präzise."
        user = f"Kennzahlen: {metrics}\nAnnahmen: {assumptions}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Kommentar (Fallback): {e}"

# =========================
#   Endpoints
# =========================

@app.get("/")
def root():
    return {"message": "Hello, Real Estate Analyst 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = (req.message or "").strip()
    params = _extract_params(text)

    # Neue Analyse
    if "price_eur" in params and "monthly_rent_eur" in params:
        ctx = req.context or {}
        res = _analyze(
            price_eur=params["price_eur"],
            monthly_rent_eur=params["monthly_rent_eur"],
            equity_ratio=params.get("equity_ratio", ctx.get("equity_ratio", 0.3)),
            interest_now=params.get("interest_now", ctx.get("interest_now", 0.03)),
            scenario_rate_delta=params.get("scenario_rate_delta", ctx.get("scenario_rate_delta", 0.01)),
            amort_rate=params.get("amort_rate", ctx.get("amort_rate", 0.02)),
            non_recoverable_costs_rate=params.get("non_recoverable_costs_rate", ctx.get("non_recoverable_costs_rate", 0.20)),
            vacancy_rate=params.get("vacancy_rate", ctx.get("vacancy_rate", 0.05)),
            acquisition_costs_rate=params.get("acquisition_costs_rate", ctx.get("acquisition_costs_rate", 0.08)),
        )
        assumptions = {**ctx, **params}
        comment = _make_commentary(res, assumptions)
        reply = (
            "Analyse durchgeführt:\n"
            f"- Bruttorendite: {res['gross_yield_pct']}%\n"
            f"- Nettorendite: {res['net_yield_pct']}%\n"
            f"- DSCR (Szenario): {res['scenario']['dscr']}\n\n"
            f"{comment}"
        )
        return ChatResponse(reply=reply, context_out={**assumptions, **res})

    # Anpassung: nur Miete
    if "monthly_rent_eur" in params and "price_eur" not in params:
        ctx = req.context or {}
        if "price_eur" in ctx:
            res = _analyze(
                price_eur=ctx["price_eur"],
                monthly_rent_eur=params["monthly_rent_eur"],
                equity_ratio=ctx.get("equity_ratio", 0.3),
                interest_now=ctx.get("interest_now", 0.03),
                scenario_rate_delta=ctx.get("scenario_rate_delta", 0.01),
            )
            new_ctx = {**ctx, **params, **res}
            reply = (
                "Miete aktualisiert.\n"
                f"- Bruttorendite: {res['gross_yield_pct']}%\n"
                f"- Nettorendite: {res['net_yield_pct']}%\n"
                f"- DSCR (Szenario): {res['scenario']['dscr']}"
            )
            return ChatResponse(reply=reply, context_out=new_ctx)

    # Anpassung: nur Preis
    if "price_eur" in params and "monthly_rent_eur" not in params:
        ctx = req.context or {}
        if "monthly_rent_eur" in ctx:
            res = _analyze(
                price_eur=params["price_eur"],
                monthly_rent_eur=ctx["monthly_rent_eur"],
                equity_ratio=ctx.get("equity_ratio", 0.3),
                interest_now=ctx.get("interest_now", 0.03),
                scenario_rate_delta=ctx.get("scenario_rate_delta", 0.01),
            )
            new_ctx = {**ctx, **params, **res}
            reply = (
                "Preis aktualisiert.\n"
                f"- Bruttorendite: {res['gross_yield_pct']}%\n"
                f"- Nettorendite: {res['net_yield_pct']}%\n"
                f"- DSCR (Szenario): {res['scenario']['dscr']}"
            )
            return ChatResponse(reply=reply, context_out=new_ctx)

    # Standardantwort
    if not USE_OPENAI:
        return ChatResponse(reply=f"Analyst (Dummy): „{text}“", context_out=req.context)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        system = "Du bist ein vorsichtiger Immobilien-Analyst (DACH)."
        ctx = f"\nKontext: {req.context}" if req.context else ""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":text+ctx}],
            temperature=0.2,
        )
        return ChatResponse(reply=resp.choices[0].message.content.strip(), context_out=req.context)
    except Exception as e:
        return ChatResponse(reply=f"Analyst (Fallback): {e}", context_out=req.context)
