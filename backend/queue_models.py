"""Pydantic models for the job queue system."""
from typing import Optional

from pydantic import BaseModel, ConfigDict


class QueueJobRequest(BaseModel):
    """Single inference job submission."""
    model: str
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    tools: Optional[list[dict]] = None  # OpenAI-format tool definitions
    metadata: Optional[dict] = None  # app-specific passthrough


class QueueBatchRequest(BaseModel):
    """Batch of inference jobs."""
    jobs: list[QueueJobRequest]


class QueueJobResponse(BaseModel):
    job_id: str
    status: str  # queued|loading_model|running|completed|failed|cancelled|waiting_for_eviction
    model: str
    position: Optional[int] = None
    warning: Optional[str] = None
    evicting: Optional[list[str]] = None


class QueueBatchResponse(BaseModel):
    batch_id: str
    jobs: list[QueueJobResponse]


class QueueJobResult(BaseModel):
    job_id: str
    status: str
    model: str
    result: Optional[dict] = None
    error: Optional[str] = None
    retried: int = 0
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Optional[dict] = None


class QueueBatchStatus(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    queued: int
    jobs: list[QueueJobResult]


class QueueOverview(BaseModel):
    queue_depth: int
    models_queued: list[str]
    models_loaded: list[str]
    current_job: Optional[str] = None
    gpu_vram_total_gb: float = 0
    gpu_vram_used_gb: float = 0
    gpu_vram_free_gb: float = 0


class ModelSettingsUpdate(BaseModel):
    do_not_evict: Optional[bool] = None
    evictable: Optional[bool] = None
    wait_for_completion: Optional[bool] = None
    vram_estimate_gb: Optional[float] = None
    categories: Optional[list[str]] = None
    safety: Optional[str] = None


class ModelSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    do_not_evict: bool = False
    evictable: bool = True
    wait_for_completion: bool = True
    vram_estimate_gb: Optional[float] = None
    categories: list[str] = []
    safety: str = "safe"


class ModelAlias(BaseModel):
    id: int
    alias_name: str
    base_model: str
    system_prompt: Optional[str] = None
    parameters: dict = {}
    description: str = ""


class ModelAliasCreate(BaseModel):
    base_model: str
    system_prompt: Optional[str] = None
    parameters: dict = {}
    description: str = ""


class ModelRunnerParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    runner_id: int
    hostname: Optional[str] = None
    system_prompt: Optional[str] = None
    parameters: dict = {}
    do_not_evict: bool = False


class ModelRunnerParamsUpsert(BaseModel):
    system_prompt: Optional[str] = None
    parameters: dict = {}
    do_not_evict: Optional[bool] = None


class EvictionError(BaseModel):
    error: str  # model_too_large | insufficient_vram
    message: str
    vram_required_gb: float
    vram_available_gb: float
    non_evictable_gb: Optional[float] = None
    loaded_models: Optional[list[dict]] = None
