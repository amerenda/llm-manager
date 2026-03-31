// Per-runner status
export interface RunnerStatus {
  runner_id: number
  runner_hostname: string
  error?: string
  gpu_vram_used_gb: number
  gpu_vram_total_gb: number
  gpu_vram_pct?: number
  cpu_pct?: number
  mem_used_gb?: number
  mem_total_gb?: number
  disk_total_gb?: number
  disk_used_gb?: number
  disk_free_gb?: number
  disk_path?: string
  loaded_ollama_models?: { name: string; size_gb: number }[]
  comfyui_running?: boolean
  gpu_vendor?: string
}

// Aggregated status from GET /api/llm/status
export interface LlmStatus {
  node?: string
  runners?: RunnerStatus[]
  gpu_vram_used_gb: number
  gpu_vram_total_gb: number
  gpu_vram_pct: number
  cpu_pct: number
  mem_used_gb: number
  mem_total_gb: number
  loaded_ollama_models: { name: string; size_gb: number; runner?: string }[]
  comfyui_running?: boolean
  comfyui_checkpoints?: string[]
  comfyui_active_checkpoint?: string | null
}

// A model entry from GET /api/llm/models
export interface LlmModel {
  id: string
  type: 'text' | 'image'
  owned_by: string
  created: number
  runners?: { runner_id: number; hostname: string }[]
  parameter_count?: string
  quantization?: string
  vram_estimate_gb?: number
  size_gb?: number
}

// A registered GPU/TTS runner from GET /api/runners
export interface Runner {
  id: number
  hostname: string
  address: string
  port: number
  enabled: boolean
  auto_update: boolean
  capabilities: {
    // GPU runner fields
    gpu_vram_total_bytes?: number
    gpu_vram_used_bytes?: number
    gpu_vram_free_bytes?: number
    disk_total_bytes?: number
    disk_used_bytes?: number
    disk_free_bytes?: number
    loaded_models?: string[]
    comfyui_running?: boolean
    agent_version?: string
    auto_update?: boolean
    // TTS runner fields
    tts?: boolean
    stt?: boolean
    voices?: string[]
    default_voice?: string
    gpu?: boolean
  }
  last_seen: string | null
  created_at: string
}

// A registered application from GET /api/apps
export interface RegisteredApp {
  id: number
  name: string
  base_url: string
  status: string             // 'pending' | 'active'
  allow_profile_switch: boolean
  allowed_models: string[]   // empty = unrestricted
  allowed_runner_ids: number[] // empty = any runner
  api_key_preview?: string
  last_seen: string | null   // ISO timestamp or null
  metadata: Record<string, unknown>
}

// ── Profiles ────────────────────────────────────────────────────────────────

export interface Profile {
  id: number
  name: string
  is_default: boolean
  unsafe_enabled: boolean
  model_entries: ProfileModelEntry[]
  image_entries: ProfileImageEntry[]
  model_entry_count?: number
  image_entry_count?: number
  created_at: string
  updated_at: string
}

export interface ProfileModelEntry {
  id: number
  profile_id: number
  model_safe: string
  model_unsafe: string | null
  count: number
  label: string | null
  parameters: Record<string, unknown>
  sort_order: number
}

export interface ProfileImageEntry {
  id: number
  profile_id: number
  checkpoint_safe: string
  checkpoint_unsafe: string | null
  label: string | null
  parameters: Record<string, unknown>
  sort_order: number
}

export interface ProfileActivation {
  runner_id: number
  profile_id: number | null
  profile_name: string | null
  activation_status: string
  activated_at: string
}

// Moltbook agent from GET /api/agents
export interface AgentPersona {
  name: string
  description: string
  tone: string
  topics: string[]
}

export interface AgentSchedule {
  post_interval_minutes: number
  active_hours_start: number
  active_hours_end: number
}

export interface AgentBehavior {
  max_post_length: number
  auto_reply: boolean
  auto_like: boolean
  reply_to_own_threads: boolean
  post_jitter_pct: number
  karma_throttle: boolean
  karma_throttle_threshold: number
  karma_throttle_multiplier: number
  target_submolts: string[]
  auto_dm_approve: boolean
  receive_peer_likes: boolean
  receive_peer_comments: boolean
  send_peer_likes: boolean
  send_peer_comments: boolean
}

export interface AgentState {
  slot: number
  karma: number
  last_heartbeat: string | null
  last_post_time: number
  next_post_time: number
  pending_dm_requests: string[]
}

export interface Agent {
  slot: number
  enabled: boolean
  model: string
  registered: boolean
  claimed: boolean
  running: boolean
  persona: AgentPersona
  schedule: AgentSchedule
  behavior: AgentBehavior
  state: AgentState
}

export interface LibraryModel {
  name: string
  description: string
  pulls: string
  parameter_sizes: string[]
  categories: string[]
  safety: string
  downloaded: boolean
  downloaded_on: string[]
  loaded: boolean
  fits: boolean
  fits_on: { runner: string; vram_total_gb: number }[]
  vram_estimate_gb: number
  size_info?: Record<string, { vram_gb: number; fits: boolean }>
}

export interface SafetyTag {
  id: number
  pattern: string
  classification: string
  reason: string
}
