"""
FastAPI application for spectral reasoning backend.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from icm_tokenizer import ICMTokenizer
from matrix_builder import MatrixBuilder
from spectral import SpectralAnalyzer
from mode_router import ModeRouter, MODES


# Request/Response schemas
class AnalyzeRequest(BaseModel):
    ask: str


class SpectralAnalysisResponse(BaseModel):
    coherence: float  # 0-100
    dominant_mode: str
    mode_activation: list[float]  # 12-element array
    lambda_1: float
    lambda_2: float
    spectral_gap: float
    debug: dict  # Internal analysis data


# Initialize
app = FastAPI(title="Spectral Backend", version="0.1.0")
tokenizer = ICMTokenizer()
matrix_builder = MatrixBuilder()
spectral_analyzer = SpectralAnalyzer()
mode_router = ModeRouter()


@app.post("/analyze", response_model=SpectralAnalysisResponse)
async def analyze_ask(request: AnalyzeRequest) -> SpectralAnalysisResponse:
    """
    Analyze an ask and return spectral coherence + mode activation.

    Example request:
    ```
    POST /analyze
    {
        "ask": "I want to build an AI that understands coherence for research teams"
    }
    ```

    Example response:
    ```
    {
        "coherence": 72.5,
        "dominant_mode": "flow",
        "mode_activation": [0.08, 0.05, 0.12, 0.25, 0.15, 0.08, 0.05, 0.12, 0.03, 0.04, 0.02, 0.01],
        "lambda_1": 4.2,
        "lambda_2": 2.1,
        "spectral_gap": 2.1,
        "debug": {...}
    }
    """
    ask = request.ask.strip()
    if not ask:
        raise HTTPException(status_code=400, detail="ask cannot be empty")

    # Step 1: Tokenize
    try:
        tokens = tokenizer.tokenize(ask)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization error: {str(e)}")

    # Step 2: Build matrix
    try:
        matrix, node_names = matrix_builder.build(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matrix building error: {str(e)}")

    # Step 3: Spectral analysis
    try:
        spectral_result = spectral_analyzer.analyze(matrix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spectral analysis error: {str(e)}")

    # Step 4: Mode routing
    try:
        mode_result = mode_router.activate(tokens, spectral_result["eigenvector_1"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mode routing error: {str(e)}")

    # Assemble response
    return SpectralAnalysisResponse(
        coherence=spectral_result["coherence"],
        dominant_mode=mode_result["dominant_mode"],
        mode_activation=mode_result["mode_activation"],
        lambda_1=spectral_result["lambda_1"],
        lambda_2=spectral_result["lambda_2"],
        spectral_gap=spectral_result["spectral_gap"],
        debug={
            "num_tokens": len(tokens),
            "matrix_size": spectral_result["matrix_size"],
            "mode_entropy": mode_result["mode_entropy"],
            "node_names": node_names[:10],  # First 10 for brevity
            "token_types": [t.group_type for t in tokens[:10]],
        },
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/modes")
async def list_modes():
    """List all 12 phronesis modes."""
    return {"modes": MODES}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
