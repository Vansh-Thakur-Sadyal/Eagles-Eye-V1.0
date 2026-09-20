/** Display helpers. Kept in one place so every screen speaks the same language. */

export const SEVERITY_ORDER = ['info', 'low', 'medium', 'high', 'critical'] as const
export type Severity = (typeof SEVERITY_ORDER)[number]

/** Semantic telemetry tones from the design system. */
export const SEVERITY_STYLE: Record<string, { pill: string; pip: string; text: string }> = {
  critical: { pill: 'bg-red-50 border-red-200 text-red-800', pip: 'bg-red-600', text: 'text-red-700' },
  high: { pill: 'bg-amber-50 border-amber-300 text-amber-900', pip: 'bg-amber-600', text: 'text-amber-700' },
  medium: { pill: 'bg-blue-50 border-blue-200 text-blue-900', pip: 'bg-blue-600', text: 'text-blue-700' },
  low: { pill: 'bg-emerald-50 border-emerald-200 text-emerald-800', pip: 'bg-emerald-600', text: 'text-emerald-700' },
  info: { pill: 'bg-surface-container border-outline-variant text-on-surface-variant', pip: 'bg-outline', text: 'text-on-surface-variant' },
}

export const STATUS_STYLE: Record<string, { pill: string; pip: string }> = {
  online: { pill: 'bg-emerald-50 border-emerald-200 text-emerald-800', pip: 'bg-emerald-600' },
  offline: { pill: 'bg-surface-container border-outline-variant text-on-surface-variant', pip: 'bg-outline' },
  degraded: { pill: 'bg-amber-50 border-amber-300 text-amber-900', pip: 'bg-amber-600' },
  error: { pill: 'bg-red-50 border-red-200 text-red-800', pip: 'bg-red-600' },
  connecting: { pill: 'bg-blue-50 border-blue-200 text-blue-900', pip: 'bg-blue-600' },
  paused: { pill: 'bg-surface-container-high border-outline-variant text-on-surface', pip: 'bg-outline' },
  disabled: { pill: 'bg-surface-container border-outline-variant text-outline', pip: 'bg-outline' },
}

export function severityStyle(severity?: string | null) {
  return SEVERITY_STYLE[(severity ?? 'info').toLowerCase()] ?? SEVERITY_STYLE.info
}

export function statusStyle(status?: string | null) {
  return STATUS_STYLE[(status ?? 'offline').toLowerCase()] ?? STATUS_STYLE.offline
}

export function titleCase(value?: string | null): string {
  if (!value) return ''
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim()
}

export function timeOf(iso?: string | null): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
}

export function dateTimeOf(iso?: string | null): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString([], {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  })
}

export function relativeTime(iso?: string | null): string {
  if (!iso) return '—'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '—'
  const seconds = Math.round((Date.now() - then) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export function duration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined) return '—'
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

export function percent(value?: number | null, digits = 0): string {
  if (value === null || value === undefined) return '—'
  return `${value.toFixed(digits)}%`
}

export function confidence(value?: number | null): string {
  if (value === null || value === undefined) return '—'
  return `${Math.round(value * 100)}%`
}

export function number(value?: number | null, digits = 0): string {
  if (value === null || value === undefined) return '—'
  return value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function bytes(value?: number | null): string {
  if (!value) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

/** Short, human names for the behaviours the agents emit. */
export const BEHAVIOR_LABEL: Record<string, string> = {
  loitering: 'Loitering',
  counter_flow: 'Counter-flow',
  running: 'Running',
  sudden_movement: 'Sudden movement',
  abnormal_trajectory: 'Erratic movement',
  restricted_entry: 'Restricted entry',
  restricted_zone_entry: 'Restricted entry',
  following: 'Possible following',
  unattended_object: 'Unattended object',
  object_separation: 'Object separated',
  object_reclaimed: 'Object reclaimed',
  crowd_surge: 'Crowd surge',
  crowd_reversal: 'Crowd reversal',
  crowd_density_critical: 'Critical density',
  crowd_flow_anomaly: 'Flow anomaly',
  watchlist_match: 'Watchlist sighting',
  face_unavailable: 'Face not observable',
  appearance_change: 'Appearance change',
  cross_camera_association: 'Cross-camera match',
  violence_detected: 'Possible violence',
  anomalous_activity: 'Anomalous activity',
  fire_smoke: 'Smoke or fire',
  line_crossing: 'Line crossing',
}

export function behaviorLabel(key?: string | null): string {
  if (!key) return '—'
  return BEHAVIOR_LABEL[key] ?? titleCase(key)
}

export const BEHAVIOR_ICON: Record<string, string> = {
  loitering: 'hourglass_top',
  counter_flow: 'sync_alt',
  running: 'directions_run',
  sudden_movement: 'bolt',
  abnormal_trajectory: 'moving',
  restricted_entry: 'block',
  restricted_zone_entry: 'block',
  following: 'follow_the_signs',
  unattended_object: 'luggage',
  object_separation: 'work_off',
  crowd_surge: 'groups',
  crowd_reversal: 'swap_horiz',
  crowd_density_critical: 'density_medium',
  crowd_flow_anomaly: 'stream',
  watchlist_match: 'person_search',
  face_unavailable: 'visibility_off',
  appearance_change: 'change_circle',
  cross_camera_association: 'link',
  violence_detected: 'report',
  anomalous_activity: 'crisis_alert',
  fire_smoke: 'local_fire_department',
}

export function behaviorIcon(key?: string | null): string {
  if (!key) return 'help'
  return BEHAVIOR_ICON[key] ?? 'sensors'
}

export function riskBand(score: number): Severity {
  if (score >= 90) return 'critical'
  if (score >= 75) return 'high'
  if (score >= 50) return 'medium'
  if (score >= 25) return 'low'
  return 'info'
}

export function densityColor(band: string): string {
  return (
    { low: 'bg-emerald-500', medium: 'bg-blue-500', high: 'bg-amber-500', critical: 'bg-red-600' }[band] ??
    'bg-outline'
  )
}
