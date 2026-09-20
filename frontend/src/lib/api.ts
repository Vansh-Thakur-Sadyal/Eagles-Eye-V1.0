/**
 * Typed API client.
 *
 * Every screen reads through this module - there is no hardcoded sample data
 * anywhere in the UI. The base URL comes from the environment so the same build
 * works behind a dev proxy, on a VPS or from the FastAPI static mount.
 */

const BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
const TOKEN_KEY = 'sentinel.token'

export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) {
    super(message)
    this.name = 'ApiError'
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* private-mode browsers: the session simply does not persist */
  }
}

type Options = RequestInit & { params?: Record<string, unknown> }

async function request<T>(path: string, options: Options = {}): Promise<T> {
  const { params, ...init } = options
  let url = `${BASE}${path}`

  if (params) {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === '') continue
      search.append(key, String(value))
    }
    const qs = search.toString()
    if (qs) url += `?${qs}`
  }

  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(url, { ...init, headers })

  if (response.status === 401) {
    setToken(null)
    if (!location.pathname.startsWith('/login')) {
      location.href = '/login'
    }
    throw new ApiError(401, 'Session expired. Please sign in again.')
  }

  const text = await response.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }

  if (!response.ok) {
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`
    throw new ApiError(response.status, detail, body)
  }
  return body as T
}

export const api = {
  get: <T,>(path: string, params?: Record<string, unknown>) =>
    request<T>(path, { method: 'GET', params }),
  post: <T,>(path: string, body?: unknown, params?: Record<string, unknown>) =>
    request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body), params }),
  put: <T,>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PUT', body: JSON.stringify(body ?? {}) }),
  patch: <T,>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body ?? {}) }),
  delete: <T,>(path: string, params?: Record<string, unknown>) =>
    request<T>(path, { method: 'DELETE', params }),
  upload: <T,>(path: string, form: FormData) =>
    request<T>(path, { method: 'POST', body: form }),
  base: BASE,
}

/** Stream URLs need the raw path: <img> cannot carry an Authorization header. */
export const streamUrl = (cameraId: string) => `${BASE}/api/cameras/${cameraId}/stream`
export const snapshotUrl = (cameraId: string) => `${BASE}/api/cameras/${cameraId}/snapshot`
export const mediaUrl = (path: string) => `${BASE}/media/${path.replace(/^.*[/\\]storage[/\\]/, '')}`

export function wsUrl(topics?: string[]): string {
  const token = getToken() ?? ''
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  const host = BASE ? BASE.replace(/^https?:\/\//, '') : location.host
  const query = new URLSearchParams({ token })
  if (topics?.length) query.set('topics', topics.join(','))
  return `${scheme}://${host}/ws/stream?${query.toString()}`
}

// ------------------------------------------------------------------ types
export interface Paged<T> {
  items: T[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface SessionUser {
  id: string
  username: string
  full_name: string
  role: string
  badge?: string | null
  post?: string | null
  permissions: string[]
}

export interface GpuStatus {
  requested_mode: string
  gpu_enabled: boolean
  cuda_available: boolean
  active_device: string
  half_precision: boolean
  gpu_index: number
  torch_available: boolean
  torch_version: string | null
  cuda_version: string | null
  gpus: { index: number; name: string; total_memory_mb: number; compute_capability: string; driver: string }[]
  memory: Partial<{
    free_mb: number
    total_mb: number
    used_mb: number
    allocated_mb: number
    utilisation_percent: number
  }>
  loaded_models: string[]
  workers: { node_id: string; name: string; gpu_name: string; online: boolean; jobs_completed: number; total_memory_mb: number }[]
  client_acceleration?: { webgpu_recommended: boolean; power_preference: string; note: string }
  warning?: string
}

export interface Camera {
  id: string
  name: string
  location: string
  source_type: string
  source_uri: string
  site_id: string | null
  zone_id: string | null
  status: string
  status_detail: string | null
  latitude: number | null
  longitude: number | null
  floor: number
  orientation_deg: number
  field_of_view_deg: number
  range_m: number
  fps: number
  width: number
  height: number
  rotation: number
  active: boolean
  running: boolean
  privacy_redaction: boolean | null
  detection_classes: string[]
  enabled_agents: string[]
  meta: Record<string, unknown>
  runtime: null | {
    status: string
    measured_fps: number
    frame_index: number
    inference_ms: number
    track_count: number
    device: string
    detector: { name: string; backend: string; ready: boolean } | null
    source: Record<string, unknown> | null
    resolution: [number, number]
    status_detail?: string | null
    /** Present when the camera runs on an edge GPU node rather than this server. */
    processing_node?: ProcessingNode | null
    processing_gpu?: string
  }
  source_help?: string
}

export interface ProcessingNode {
  node_id: string
  name: string
  gpu_name: string
  online: boolean
}

export interface EdgeNode {
  node_id: string
  name: string
  device: string
  gpu_name: string
  total_memory_mb: number
  max_cameras: number
  online: boolean
  seconds_since_seen: number
  cameras: string[]
  frames_received: number
  events_received: number
  incidents_received: number
  software: Record<string, unknown>
}

export interface EdgeStatus {
  placement: 'auto' | 'server' | 'edge'
  nodes: EdgeNode[]
  online: number
  remote_cameras: number
}

export interface Incident {
  id: string
  camera_id: string | null
  zone_id: string | null
  event_type: string
  title: string
  summary: string | null
  explanation: string | null
  risk_score: number
  severity: string
  status: string
  confidence: number
  risk_factors: RiskFactor[]
  recommended_actions: RecommendedAction[]
  track_ids: string[]
  object_ids: string[]
  started_at: string
  last_update_at: string
  location: Record<string, unknown>
  meta: Record<string, unknown>
  assigned_to?: string | null
  timeline?: TimelineEntry[]
}

export interface RiskFactor {
  behavior: string
  weight: number
  confidence: number
  recency_factor: number
  contribution: number
  severity: string
  explanation: string
  counted: boolean
  note: string | null
  track_id?: string | null
  zone_id?: string | null
}

export interface RecommendedAction {
  key: string
  label: string
  priority: number
  requires_human_approval: boolean
  consequence: string
  status: string
  rationale: string
}

export interface TimelineEntry {
  id: string
  at: string
  kind: string
  actor: string
  text: string
  confidence: number | null
  payload: Record<string, unknown>
}

export interface BehaviorEventRow {
  id: string
  camera_id: string
  zone_id: string | null
  track_id: string | null
  secondary_track_id: string | null
  behavior: string
  severity: string
  confidence: number
  started_at: string
  duration_seconds: number
  explanation: string | null
  evidence: Record<string, unknown>
  agent: string
}

export interface CrowdRow {
  camera_id: string
  camera_name: string | null
  zone_id: string | null
  count: number
  density: number
  density_band: string
  risk: string
  mean_speed: number
  flow_direction_deg: number | null
  counter_flow_ratio: number
  compression: number
  delta_percent: number
}

export interface LiveTrack {
  track_id: number
  global_id: string | null
  class_name: string
  bbox: number[]
  score: number
  speed: number
  heading_deg: number | null
  zone_id: string | null
  age_frames: number
  face_visibility: string
  duration_seconds: number
}

export interface WatchlistSubject {
  id: string
  label: string
  category: string
  priority: string
  status: string
  description: string | null
  legal_basis: string
  case_reference: string | null
  requested_by: string | null
  approved_by: string | null
  approved_at: string | null
  expires_at: string | null
  match_threshold: number | null
  alert_on_match: boolean
  scope_camera_ids: string[]
  face_embedding_count: number
  body_embedding_count: number
  augmentation_count: number
  match_count: number
  last_match_at: string | null
  matching_active: boolean
  expired: boolean
  total_sightings: number
  reference_image_count: number
  augmented_image_count: number
  images: { id: string; kind: string; augmentation: string | null; file_path: string; quality_score: number }[]
}

export interface AgentRow {
  name: string
  spec_id: number
  description: string
  enabled: boolean
  location: string
  backend?: string | null
  last_run_ms?: number
  total_runs?: number
  total_findings?: number
  last_error?: string | null
}
