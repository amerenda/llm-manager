// GPU / system stats returned by GET /api/llm/status
export interface LlmStatus {
  node: string
  gpu_vram_used_gb: number
  gpu_vram_total_gb: number
  gpu_vram_pct: number
  cpu_pct: number
  mem_used_gb: number
  mem_total_gb: number
  loaded_ollama_models: { name: string; size_gb: number }[]
  comfyui_running: boolean
  comfyui_checkpoints: string[]
  comfyui_active_checkpoint: string | null
}

// A model entry from GET /api/llm/models
export interface LlmModel {
  id: string
  type: 'text' | 'image'
  owned_by: string
  created: number
}

// A registered GPU/TTS runner from GET /api/runners
export interface Runner {
  id: number
  hostname: string
  address: string
  port: number
  capabilities: {
    // GPU runner fields
    gpu_vram_total_bytes?: number
    gpu_vram_used_bytes?: number
    gpu_vram_free_bytes?: number
    loaded_models?: string[]
    comfyui_running?: boolean
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
  id: string
  name: string
  base_url: string
  last_seen: string | null   // ISO timestamp or null
  metadata: Record<string, unknown>
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
