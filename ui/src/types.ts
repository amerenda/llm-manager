// GPU / system stats returned by GET /api/llm/status
export interface LlmStatus {
  gpu: {
    vram_used_gb: number
    vram_total_gb: number
    vram_pct: number
    name: string
  }
  cpu_pct: number
  mem_used_gb: number
  mem_total_gb: number
  loaded_ollama_models: { name: string; size_gb: number }[]
  comfyui_running: boolean
  comfyui_checkpoints: string[]
  comfyui_active_checkpoint: string | null
  ollama_reachable: boolean
}

// A model entry from GET /api/llm/models
export interface LlmModel {
  id: string
  name: string
  type: 'text' | 'image'
  size_gb: number
  is_loaded: boolean
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
