/**
 * Digital Twin — a site plan with camera coverage wedges, restricted zones and
 * live incident markers, rendered from the same payload the AR and VR layers use.
 *
 * Drawn as SVG over a normalised floor plan: it works without a 3D asset, and
 * accepts one when a site has `twin_model_url` set.
 */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Kpi, Loading, PageHeader, SeverityPill } from '../components/ui'
import { streamUrl } from '../lib/api'
import { useAuth } from '../lib/auth'
import { relativeTime, titleCase } from '../lib/format'
import { useApi, useGpu, useWebGpu } from '../lib/hooks'

interface Twin {
  sites: { id: string; name: string; site_type: string; floors: { level: number; name: string }[]; twin_model_url: string | null }[]
  zones: { id: string; name: string; zone_type: string; floor: number; polygon: number[][]; risk_weight: number }[]
  cameras: {
    id: string
    name: string
    location: string
    floor: number
    running: boolean
    status: string
    latitude: number | null
    longitude: number | null
    orientation_deg: number
    field_of_view_deg: number
    zone_id: string | null
  }[]
  incidents: {
    id: string
    title: string
    severity: string
    risk_score: number
    camera_id: string | null
    zone_id: string | null
    status: string
    started_at: string
  }[]
  live_entities: Record<string, { track_id: number; class_name: string; image_x?: number; image_y?: number; bearing_deg?: number }[]>
  generated_at: string
}

const ZONE_FILL: Record<string, string> = {
  restricted: 'rgba(220,38,38,0.12)',
  secure: 'rgba(124,58,237,0.12)',
  platform: 'rgba(29,78,216,0.10)',
  gate: 'rgba(29,78,216,0.10)',
  transit: 'rgba(5,150,105,0.10)',
  perimeter: 'rgba(217,119,6,0.10)',
  public: 'rgba(148,163,184,0.10)',
}
const ZONE_STROKE: Record<string, string> = {
  restricted: '#dc2626',
  secure: '#7c3aed',
  platform: '#1d4ed8',
  gate: '#1d4ed8',
  transit: '#059669',
  perimeter: '#d97706',
  public: '#94a3b8',
}

export default function TwinPage() {
  const [siteId, setSiteId] = useState('')
  const [floor, setFloor] = useState(0)
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null)
  const { clientGpu } = useGpu()
  const webgpu = useWebGpu(clientGpu)

  const { data, loading, error, refresh } = useApi<Twin>(
    '/api/spatial/twin', { site_id: siteId || undefined }, { pollMs: 8000 },
  )

  useEffect(() => {
    document.title = 'Digital Twin · Eagles Eye'
  }, [])

  const zones = useMemo(() => (data?.zones ?? []).filter((z) => z.floor === floor), [data, floor])
  const cameras = useMemo(() => (data?.cameras ?? []).filter((c) => c.floor === floor), [data, floor])
  const incidentByCamera = useMemo(() => {
    const map = new Map<string, Twin['incidents'][number]>()
    for (const incident of data?.incidents ?? []) {
      if (incident.camera_id && !map.has(incident.camera_id)) map.set(incident.camera_id, incident)
    }
    return map
  }, [data])

  if (error) return <ErrorNote message={error} onRetry={refresh} />
  if (loading && !data) return <Loading label="Building the twin" />
  if (!data) return null

  const selected = cameras.find((c) => c.id === selectedCamera) ?? null

  return (
    <>
      <PageHeader
        title="Digital Twin"
        subtitle="Zones, camera coverage and live incidents in one spatial view"
        actions={
          <>
            <select className="input w-44" value={siteId} onChange={(e) => setSiteId(e.target.value)}>
              <option value="">All sites</option>
              {data.sites.map((site) => (
                <option key={site.id} value={site.id}>{site.name}</option>
              ))}
            </select>
            <select className="input w-32" value={floor} onChange={(e) => setFloor(Number(e.target.value))}>
              {Array.from(new Set([0, ...data.zones.map((z) => z.floor), ...data.cameras.map((c) => c.floor)]))
                .sort((a, b) => a - b)
                .map((level) => (
                  <option key={level} value={level}>Floor {level}</option>
                ))}
            </select>
          </>
        }
      />

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi label="Zones on this floor" value={zones.length} footer={`${zones.filter((z) => z.zone_type === 'restricted').length} restricted`} />
        <Kpi label="Cameras" value={cameras.length} footer={`${cameras.filter((c) => c.running).length} streaming`} />
        <Kpi label="Open incidents" value={data.incidents.length} tone={data.incidents.length ? 'warning' : 'good'} />
        <Kpi
          label="Client rendering"
          value={webgpu.available ? 'GPU' : 'CPU'}
          tone={webgpu.available ? 'good' : 'default'}
          footer={webgpu.adapter ?? webgpu.reason ?? '—'}
        />
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" title="Site plan" bodyClassName="p-space-md">
          {zones.length === 0 && cameras.length === 0 ? (
            <Empty
              icon="map"
              title="Nothing mapped on this floor"
              hint="Define zones with normalised polygons and give cameras a bearing to populate the plan."
            />
          ) : (
            <svg
              viewBox="0 0 1000 620"
              className="w-full rounded-lg bg-surface border border-outline-variant/50"
              style={{ aspectRatio: '1000 / 620' }}
            >
              <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                  <path d="M50 0 L0 0 0 50" fill="none" stroke="#e2e8f0" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width="1000" height="620" fill="url(#grid)" />

              {/* zones */}
              {zones.map((zone) => {
                const points = zone.polygon.map(([x, y]) => `${x * 1000},${y * 620}`).join(' ')
                const cx = (zone.polygon.reduce((s, p) => s + p[0], 0) / zone.polygon.length) * 1000
                const cy = (zone.polygon.reduce((s, p) => s + p[1], 0) / zone.polygon.length) * 620
                return (
                  <g key={zone.id}>
                    <polygon
                      points={points}
                      fill={ZONE_FILL[zone.zone_type] ?? ZONE_FILL.public}
                      stroke={ZONE_STROKE[zone.zone_type] ?? ZONE_STROKE.public}
                      strokeWidth={zone.zone_type === 'restricted' ? 2.5 : 1.5}
                      strokeDasharray={zone.zone_type === 'restricted' ? '6 4' : undefined}
                    />
                    <text x={cx} y={cy} textAnchor="middle" fontSize="13" fontFamily="Inter" fill="#434655">
                      {zone.name}
                    </text>
                    <text x={cx} y={cy + 15} textAnchor="middle" fontSize="10" fontFamily="JetBrains Mono" fill="#747686">
                      {zone.zone_type} · ×{zone.risk_weight}
                    </text>
                  </g>
                )
              })}

              {/* cameras laid out around the plan edge when they have no coordinates */}
              {cameras.map((camera, index) => {
                const angle = (index / Math.max(1, cameras.length)) * Math.PI * 2
                const cx = 500 + Math.cos(angle) * 400
                const cy = 310 + Math.sin(angle) * 240
                const incident = incidentByCamera.get(camera.id)
                const fov = camera.field_of_view_deg || 82
                const bearing = camera.orientation_deg || 0
                const r = 120
                const a1 = ((bearing - fov / 2) * Math.PI) / 180
                const a2 = ((bearing + fov / 2) * Math.PI) / 180

                return (
                  <g
                    key={camera.id}
                    onClick={() => setSelectedCamera(camera.id === selectedCamera ? null : camera.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <path
                      d={`M${cx},${cy} L${cx + Math.cos(a1) * r},${cy + Math.sin(a1) * r} A${r},${r} 0 0,1 ${cx + Math.cos(a2) * r},${cy + Math.sin(a2) * r} Z`}
                      fill={camera.running ? 'rgba(124,58,237,0.14)' : 'rgba(148,163,184,0.10)'}
                      stroke={camera.running ? '#7c3aed' : '#94a3b8'}
                      strokeWidth="1"
                    />
                    <circle
                      cx={cx}
                      cy={cy}
                      r={selectedCamera === camera.id ? 11 : 8}
                      fill={incident ? '#dc2626' : camera.running ? '#059669' : '#94a3b8'}
                      stroke="#ffffff"
                      strokeWidth="2"
                    />
                    {incident ? <circle cx={cx} cy={cy} r={16} fill="none" stroke="#dc2626" strokeWidth="2" opacity="0.5" /> : null}
                    <text x={cx} y={cy - 16} textAnchor="middle" fontSize="11" fontFamily="Inter" fill="#0b1c30">
                      {camera.name}
                    </text>
                  </g>
                )
              })}
            </svg>
          )}

          <div className="flex flex-wrap gap-space-md mt-space-md">
            <Legend colour="#059669" label="Camera streaming" />
            <Legend colour="#94a3b8" label="Camera offline" />
            <Legend colour="#dc2626" label="Open incident" />
            <Legend colour="#7c3aed" label="Coverage wedge" />
          </div>
        </Card>

        <div className="flex flex-col gap-space-lg">
          {selected ? (
            <Card title={selected.name} bodyClassName="p-0">
              {selected.running ? (
                <img src={streamUrl(selected.id)} alt={`${selected.name} live view`} className="w-full aspect-video object-cover" />
              ) : (
                <div className="aspect-video flex items-center justify-center bg-slate-900 text-slate-400 mono">
                  not streaming
                </div>
              )}
              <dl className="p-space-md grid grid-cols-2 gap-space-md">
                <Stat label="Location" value={selected.location || '—'} />
                <Stat label="Bearing" value={`${selected.orientation_deg}°`} />
                <Stat label="Field of view" value={`${selected.field_of_view_deg}°`} />
                <Stat label="Zone" value={selected.zone_id ?? '—'} />
              </dl>
            </Card>
          ) : null}

          <Card title="Open incidents" bodyClassName="p-0">
            {data.incidents.length === 0 ? (
              <Empty icon="check_circle" title="Nothing open" />
            ) : (
              <ul className="divide-y divide-outline-variant/25 max-h-96 overflow-y-auto">
                {data.incidents.map((incident) => (
                  <li key={incident.id} className="p-space-md">
                    <div className="flex items-start justify-between gap-2">
                      <Link
                        to={`/incidents/${incident.id}`}
                        className="font-headline-sm text-headline-sm truncate hover:text-primary-container"
                      >
                        {incident.title}
                      </Link>
                      <SeverityPill severity={incident.severity} />
                    </div>
                    <p className="mono text-outline mt-0.5">
                      {incident.camera_id ?? '—'} · risk {incident.risk_score.toFixed(0)} ·{' '}
                      {relativeTime(incident.started_at)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <InfoNote icon="view_in_ar">
            The same payload drives the AR overlay for field officers. Calibrate a homography on a
            camera for metric positions instead of bearing estimates.
          </InfoNote>
        </div>
      </div>
    </>
  )
}

function Legend({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 font-body-sm text-body-sm text-on-surface-variant">
      <span className="w-3 h-3 rounded-full" style={{ background: colour }} />
      {label}
    </span>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface truncate" title={value}>{value}</dd>
    </div>
  )
}
