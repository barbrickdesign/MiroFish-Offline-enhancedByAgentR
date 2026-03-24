"""
System Status API

GET /api/status — returns:
  - Neo4j connection state
  - Ollama reachability and available models
  - Disk usage for the data directory
  - Current configuration summary
"""

import os
import shutil

import requests
from flask import jsonify, current_app

from . import graph_bp
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.status')


@graph_bp.route('/status', methods=['GET'])
def system_status():
    """
    Return system health and configuration status.

    Response fields:
      neo4j    — connected (bool), uri, error (if any)
      ollama   — reachable (bool), models (list), embedding_model, llm_model
      disk     — total_gb, used_gb, free_gb, used_pct
      config   — key configuration values (no secrets)
    """
    status = {
        "neo4j": _check_neo4j(),
        "ollama": _check_ollama(),
        "disk": _check_disk(),
        "config": _config_summary(),
    }
    # Overall health: all critical services connected
    status["healthy"] = status["neo4j"]["connected"] and status["ollama"]["reachable"]
    return jsonify(status)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_neo4j() -> dict:
    """Verify Neo4j connectivity by running a trivial Cypher query."""
    result = {
        "connected": False,
        "uri": Config.NEO4J_URI,
        "error": None,
    }
    try:
        storage = current_app.extensions.get('neo4j_storage')
        if storage is None:
            result["error"] = "Neo4jStorage not initialised (check startup logs)"
            return result
        # Use the existing driver to run a no-op query
        with storage._driver.session() as session:
            session.run("RETURN 1")
        result["connected"] = True
    except Exception as e:
        result["error"] = str(e)
        logger.warning("Neo4j health check failed: %s", e)
    return result


def _check_ollama() -> dict:
    """
    Query Ollama's /api/tags endpoint to list locally available models.
    Marks whether the configured embedding and LLM models are present.
    """
    base_url = Config.EMBEDDING_BASE_URL.rstrip('/')
    result = {
        "reachable": False,
        "base_url": base_url,
        "models": [],
        "embedding_model": Config.EMBEDDING_MODEL,
        "embedding_model_available": False,
        "llm_model": Config.LLM_MODEL_NAME,
        "llm_model_available": False,
        "error": None,
    }
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        result["reachable"] = True
        result["models"] = models

        # Check if configured models are present (allow partial name match for tags)
        def _model_present(target: str, available: list) -> bool:
            target_base = target.split(":")[0].lower()
            for m in available:
                if m.lower() == target.lower() or m.lower().startswith(target_base + ":"):
                    return True
            return False

        result["embedding_model_available"] = _model_present(
            Config.EMBEDDING_MODEL, models
        )
        result["llm_model_available"] = _model_present(
            Config.LLM_MODEL_NAME, models
        )
    except requests.exceptions.ConnectionError:
        result["error"] = f"Cannot reach Ollama at {base_url}"
        logger.warning("Ollama health check: connection refused at %s", base_url)
    except Exception as e:
        result["error"] = str(e)
        logger.warning("Ollama health check failed: %s", e)
    return result


def _check_disk() -> dict:
    """Return disk usage stats for the uploads/data directory."""
    data_dir = Config.UPLOAD_FOLDER
    result = {
        "path": data_dir,
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "used_pct": None,
        "error": None,
    }
    try:
        os.makedirs(data_dir, exist_ok=True)
        usage = shutil.disk_usage(data_dir)
        gb = 1024 ** 3
        result["total_gb"] = round(usage.total / gb, 2)
        result["used_gb"] = round(usage.used / gb, 2)
        result["free_gb"] = round(usage.free / gb, 2)
        result["used_pct"] = round(usage.used / usage.total * 100, 1) if usage.total else 0
    except Exception as e:
        result["error"] = str(e)
        logger.warning("Disk check failed: %s", e)
    return result


def _config_summary() -> dict:
    """Return non-sensitive configuration values."""
    return {
        "llm_base_url": Config.LLM_BASE_URL,
        "llm_model": Config.LLM_MODEL_NAME,
        "embedding_model": Config.EMBEDDING_MODEL,
        "embedding_base_url": Config.EMBEDDING_BASE_URL,
        "neo4j_uri": Config.NEO4J_URI,
        "search_vector_weight": Config.SEARCH_VECTOR_WEIGHT,
        "search_keyword_weight": Config.SEARCH_KEYWORD_WEIGHT,
        "default_chunk_size": Config.DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
        "oasis_max_rounds": Config.OASIS_DEFAULT_MAX_ROUNDS,
    }
