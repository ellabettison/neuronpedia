"""Return SAE decoder vectors for a batch of features.

Used to reconstruct a residual-stream direction from SAE feature weights,
e.g. to build an Arditi-style refusal direction for ORTHOGONAL_DECOMP steering.
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import with_request_lock

logger = logging.getLogger(__name__)

router = APIRouter()


class DecoderVectorRequest(BaseModel):
    """Request for decoder vectors."""
    features: list[dict[str, Any]]  # [{"source": "16-gemmascope-...", "index": 1234}, ...]


class DecoderVectorResponse(BaseModel):
    """Response with decoder vectors and hook names."""
    vectors: list[dict[str, Any]]
    # [{"source": ..., "index": ..., "hook_name": ..., "vector": [float, ...]}, ...]


@router.post("/util/sae-decoder-vectors")
@with_request_lock()
async def sae_decoder_vectors(request: DecoderVectorRequest):
    """Return W_dec rows for a list of (source, index) pairs.

    Each returned vector is the decoder direction for that SAE feature in
    residual-stream space. These can be weighted and summed to reconstruct
    a combined direction (e.g. a refusal direction) for use with
    ORTHOGONAL_DECOMP steering via NPSteerVector.
    """
    sae_manager = SAEManager.get_instance()
    results = []

    for feat in request.features:
        source = feat["source"]
        index = int(feat["index"])
        try:
            sae = sae_manager.get_sae(source)
            vec = sae.W_dec[index].detach().tolist()
            hook_name = sae.cfg.metadata.hook_name
            results.append({
                "source": source,
                "index": index,
                "hook_name": hook_name,
                "vector": vec,
            })
        except Exception as e:
            logger.warning("Failed to get decoder vector for %s/%d: %s", source, index, e)
            return JSONResponse(
                content={"error": f"Failed to get decoder vector for {source}/{index}: {e}"},
                status_code=400,
            )

    return DecoderVectorResponse(vectors=results)