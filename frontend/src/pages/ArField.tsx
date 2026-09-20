/**
 * AR Field Vision — what a field officer's device would overlay.
 *
 * Uses the browser's geolocation only when the officer asks for it, so the
 * bearing and distance to each incident are real rather than assumed.
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Loading, PageHeader, SeverityPill, toast } from '../components/ui'
import { relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface ArOverlay {
  incident_id: string
  title: string
  severity: string
  risk_score: number
  anchor: { latitude: number | null; longitude: number | null; floor: number; bearing_deg: number | null }
  detail: string | null
  zone: string | null
  camera_name: string | null
  distance_m: number | null
  overlays: Record<string, boolean>
}

interface ArFeed {
  overlays: ArOverlay[]
  restricted_zones: { id: string; name: string; floor: number; polygon: number[][] }[]
  observer: { latitude: number | null; longitude: number | null; floor: number | null; radius_m: number }
}

export default function ArFieldPage() {
  const [position, setPosition] = useState<{ latitude: number; longitude: number; accuracy: number } | null>(null)
  const [radius, setRadius] = useState(500)
  const [locating, setLocating] = useState(false)

  const { data, loading, error, refresh } = useApi<ArFeed>(
    '/api/spatial/ar/feed',
    {
      latitude: position?.latitude,
      longitude: position?.longitude,
      radius_m: position ? radius : undefined,
    },
    { pollMs: 10000 },
  )

  useEffect(() => {
    document.title = 'AR Field Vision · Eagles Eye'
  }, [])

  function locate() {
    if (!navigator.geolocation) {
      toast('This browser does not expose geolocation', 'error')
      return
    }
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setPosition({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        })
        setLocating(false)
        toast('Position acquired — incidents are now sorted by distance')
      },
      (err) => {
        setLocating(false)
        toast(`Could not get a position: ${err.message}`, 'error')
      },
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }

  return (
    <>
      <PageHeader
        title="AR Field Vision"
        subtitle="Incident overlays, coverage and restricted areas for officers in the field"
        actions={
          <>
            {position ? (
              <select className="input w-36" value={radius} onChange={(e) => setRadius(Number(e.target.value))}>
                {[100, 250, 500, 1000, 5000].map((m) => (
                  <option key={m} value={m}>within {m >= 1000 ? `${m / 1000} km` : `${m} m`}</option>
                ))}
              </select>
            ) : null}
            <button className="btn-secondary" onClick={locate} disabled={locating}>
              {locating ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : <Icon name="my_location" className="text-[16px]" />}
              {position ? 'Update position' : 'Use my position'}
            </button>
          </>
        }
      />

      {!position ? (
        <InfoNote icon="location_searching">
          Without a position, every open incident is listed. Sharing your location sorts them by
          distance and filters to the chosen radius — the browser asks you first, and the position is
          used only for this query.
        </InfoNote>
      ) : (
        <InfoNote icon="my_location">
          Position {position.latitude.toFixed(5)}, {position.longitude.toFixed(5)} (±
          {position.accuracy.toFixed(0)} m). Showing incidents within {radius} m.
        </InfoNote>
      )}

      {error ? <ErrorNote message={error} onRetry={refresh} /> : null}
      {loading && !data ? <Loading /> : null}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <div className="xl:col-span-2 flex flex-col gap-space-lg">
          {!data || data.overlays.length === 0 ? (
            <Card>
              <Empty
                icon="view_in_ar"
                title="No incidents to overlay"
                hint="Overlays appear for open incidents anchored to a camera with known geometry."
              />
            </Card>
          ) : (
            data.overlays.map((overlay) => (
              <Card key={overlay.incident_id} className={overlay.severity === 'critical' ? 'ring-1 ring-red-400' : ''}>
                <div className="flex items-start justify-between gap-space-md">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <SeverityPill severity={overlay.severity} />
                      <Link
                        to={`/incidents/${overlay.incident_id}`}
                        className="font-headline-md text-headline-md hover:text-primary-container"
                      >
                        {overlay.title}
                      </Link>
                    </div>
                    <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">{overlay.detail}</p>
                    <div className="flex flex-wrap gap-space-md mt-space-md">
                      <Fact icon="videocam" label={overlay.camera_name ?? '—'} />
                      {overlay.zone ? <Fact icon="fence" label={overlay.zone} /> : null}
                      <Fact icon="layers" label={`Floor ${overlay.anchor.floor}`} />
                      {overlay.anchor.bearing_deg !== null ? (
                        <Fact icon="explore" label={`${overlay.anchor.bearing_deg.toFixed(0)}°`} />
                      ) : null}
                      {overlay.distance_m !== null ? (
                        <Fact icon="straighten" label={`${overlay.distance_m.toFixed(0)} m away`} />
                      ) : null}
                      <Fact icon="speed" label={`Risk ${overlay.risk_score.toFixed(0)}`} />
                    </div>
                  </div>

                  {overlay.distance_m !== null ? (
                    <div className="shrink-0 text-center">
                      <div
                        className="w-16 h-16 rounded-full border-2 border-outline-variant flex items-center justify-center"
                        style={{
                          transform: `rotate(${overlay.anchor.bearing_deg ?? 0}deg)`,
                        }}
                      >
                        <Icon name="navigation" className="text-[26px] text-primary-container" />
                      </div>
                      <p className="mono text-on-surface-variant mt-1">{overlay.distance_m.toFixed(0)} m</p>
                    </div>
                  ) : null}
                </div>

                <div className="flex flex-wrap gap-space-sm mt-space-md pt-space-md border-t border-outline-variant/40">
                  {Object.entries(overlay.overlays)
                    .filter(([, on]) => on)
                    .map(([key]) => (
                      <span key={key} className="pill bg-violet-50 border-violet-200 text-violet-800">
                        <Icon name="layers" className="text-[12px]" />
                        {titleCase(key)}
                      </span>
                    ))}
                </div>
              </Card>
            ))
          )}
        </div>

        <Card title="Restricted areas" bodyClassName="p-0">
          {!data || data.restricted_zones.length === 0 ? (
            <Empty icon="fence" title="No restricted areas defined" />
          ) : (
            <ul className="divide-y divide-outline-variant/25">
              {data.restricted_zones.map((zone) => (
                <li key={zone.id} className="p-space-md flex items-center gap-space-sm">
                  <Icon name="block" className="text-[18px] text-red-600 shrink-0" />
                  <div className="min-w-0">
                    <p className="font-headline-sm text-headline-sm truncate">{zone.name}</p>
                    <p className="mono text-outline">Floor {zone.floor} · {zone.polygon.length} vertices</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </>
  )
}

function Fact({ icon, label }: { icon: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 mono text-on-surface-variant">
      <Icon name={icon} className="text-[14px]" />
      {label}
    </span>
  )
}
