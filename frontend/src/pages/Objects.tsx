/** Object Intelligence — bag/person association and unattended-belonging status. */
import { useEffect, useState } from 'react'
import { Card, Empty, ErrorNote, Icon, InfoNote, Kpi, Loading, PageHeader, StatusPill, toast } from '../components/ui'
import { ApiError, api, type Paged } from '../lib/api'
import { confidence, dateTimeOf, duration, relativeTime, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface LiveObject {
  object_track_id: number
  class_name: string
  status: string
  owner_track_id: number | null
  owner_confidence: number
  stationary: boolean
  bbox: number[]
}

interface LiveCameraObjects {
  camera_id: string
  camera_name: string | null
  objects: LiveObject[]
}

interface StoredObject {
  id: string
  camera_id: string
  class_name: string
  owner_track_id: string | null
  owner_confidence: number
  first_seen: string
  last_seen: string
  separation_at: string | null
  stationary_seconds: number
  status: string
}

const STATUS_TONE: Record<string, string> = {
  attended: 'online',
  separated: 'degraded',
  warning: 'degraded',
  unattended: 'error',
  unowned: 'paused',
  reclaimed: 'online',
}

export default function ObjectsPage() {
  const [trajectory, setTrajectory] = useState<{ camera: string; object: number; points: any[] } | null>(null)

  const { data: live, loading, error, refresh } = useApi<{ cameras: LiveCameraObjects[]; note: string }>(
    '/api/intel/objects/live', undefined, { pollMs: 5000 },
  )
  const { data: stored, refresh: refreshStored } = useApi<Paged<StoredObject>>(
    '/api/intel/objects', { limit: 100 }, { pollMs: 20000 },
  )

  useLiveEvents(['finding'], (event) => {
    const behavior = event.payload?.behavior
    if (behavior === 'unattended_object' || behavior === 'object_separation' || behavior === 'object_reclaimed') {
      refresh()
      refreshStored()
    }
  })

  useEffect(() => {
    document.title = 'Object Intelligence · Eagles Eye'
  }, [])

  const all = (live?.cameras ?? []).flatMap((camera) =>
    camera.objects.map((object) => ({ ...object, camera_id: camera.camera_id, camera_name: camera.camera_name })),
  )
  const unattended = all.filter((o) => o.status === 'unattended')
  const warning = all.filter((o) => o.status === 'warning')

  async function loadTrajectory(cameraId: string, objectTrackId: number) {
    try {
      const result = await api.get<{ points: any[] }>(
        `/api/intel/objects/${cameraId}/${objectTrackId}/owner-trajectory`,
      )
      setTrajectory({ camera: cameraId, object: objectTrackId, points: result.points })
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Object Intelligence"
        subtitle="Bags and luggage, their associated subject, and how long they have been alone"
      />

      <InfoNote icon="policy">
        Eagles Eye reports association and stationarity only. It does not infer, and has no code path
        that can infer, what is inside a bag or container — that stays a physical-security task with
        a human in charge.
      </InfoNote>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi label="Tracked objects" value={all.length} footer={`${live?.cameras.length ?? 0} cameras`} />
        <Kpi label="Unattended" value={unattended.length} tone={unattended.length ? 'critical' : 'good'} />
        <Kpi label="Approaching threshold" value={warning.length} tone={warning.length ? 'warning' : 'good'} />
        <Kpi
          label="Owner associated"
          value={all.filter((o) => o.owner_track_id !== null).length}
          footer="by proximity and continuity"
        />
      </section>

      {error ? <ErrorNote message={error} onRetry={refresh} /> : null}

      <Card title="Live objects" bodyClassName="p-0">
        {loading && !live ? (
          <Loading />
        ) : all.length === 0 ? (
          <Empty
            icon="luggage"
            title="No objects being tracked"
            hint="The object agent watches the classes listed in Settings → Object policy. A detector with a trained bag class is needed for real luggage detection."
          />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Object</th>
                  <th>Camera</th>
                  <th>Associated subject</th>
                  <th>Association confidence</th>
                  <th>Stationary</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {all.map((object) => (
                  <tr key={`${object.camera_id}-${object.object_track_id}`}>
                    <td>
                      <span className="flex items-center gap-2">
                        <Icon name="luggage" className="text-[16px] text-on-surface-variant" />
                        {titleCase(object.class_name)} <span className="mono text-outline">#{object.object_track_id}</span>
                      </span>
                    </td>
                    <td className="truncate max-w-[12rem]">{object.camera_name ?? object.camera_id}</td>
                    <td className="mono">{object.owner_track_id === null ? 'none' : `track ${object.owner_track_id}`}</td>
                    <td className="mono tabular-nums">{confidence(object.owner_confidence)}</td>
                    <td className="mono">{object.stationary ? 'yes' : 'no'}</td>
                    <td><StatusPill status={STATUS_TONE[object.status] ?? 'paused'} label={object.status} /></td>
                    <td className="w-44">
                      {object.owner_track_id !== null ? (
                        <button
                          className="btn-secondary btn-xs"
                          onClick={() => loadTrajectory(object.camera_id, object.object_track_id)}
                        >
                          <Icon name="timeline" className="text-[14px]" />
                          Owner trajectory
                        </button>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {trajectory ? (
        <Card
          title={`Last associated subject · object #${trajectory.object}`}
          actions={
            <button className="btn-ghost btn-xs" onClick={() => setTrajectory(null)}>
              <Icon name="close" className="text-[16px]" />
            </button>
          }
        >
          {trajectory.points.length === 0 ? (
            <Empty icon="timeline" title="No trajectory retained" />
          ) : (
            <>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
                {trajectory.points.length} retained positions for the subject last associated with this
                object. This is an association by proximity and continuity, not an identification.
              </p>
              <TrajectoryPlot points={trajectory.points} />
            </>
          )}
        </Card>
      ) : null}

      <Card title={`Recorded objects · ${stored?.total ?? 0}`} bodyClassName="p-0">
        {!stored || stored.items.length === 0 ? (
          <Empty icon="inventory_2" title="Nothing recorded yet" />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Object</th>
                  <th>Camera</th>
                  <th>Owner track</th>
                  <th>First seen</th>
                  <th>Separated</th>
                  <th>Stationary</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {stored.items.map((row) => (
                  <tr key={row.id}>
                    <td>{titleCase(row.class_name)}</td>
                    <td className="mono truncate max-w-[10rem]">{row.camera_id}</td>
                    <td className="mono">{row.owner_track_id ?? '—'}</td>
                    <td className="mono" title={dateTimeOf(row.first_seen)}>{relativeTime(row.first_seen)}</td>
                    <td className="mono">{row.separation_at ? relativeTime(row.separation_at) : '—'}</td>
                    <td className="mono">{duration(row.stationary_seconds)}</td>
                    <td><StatusPill status={STATUS_TONE[row.status] ?? 'paused'} label={row.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}

function TrajectoryPlot({ points }: { points: { x: number; y: number; t?: string }[] }) {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const spanX = Math.max(1, maxX - minX)
  const spanY = Math.max(1, maxY - minY)

  const path = points
    .map((p, i) => {
      const x = ((p.x - minX) / spanX) * 940 + 20
      const y = ((p.y - minY) / spanY) * 300 + 20
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  const last = points[points.length - 1]
  const lastX = ((last.x - minX) / spanX) * 940 + 20
  const lastY = ((last.y - minY) / spanY) * 300 + 20

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox="0 0 980 340" className="w-full min-w-[520px] h-64 rounded-lg bg-surface border border-outline-variant/40">
        <path d={path} fill="none" stroke="#1d4ed8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={((points[0].x - minX) / spanX) * 940 + 20} cy={((points[0].y - minY) / spanY) * 300 + 20} r={5} fill="#059669" />
        <circle cx={lastX} cy={lastY} r={5} fill="#dc2626" />
        <text x={16} y={330} className="mono" fontSize={11} fill="#434655">
          green = first retained position · red = last retained position
        </text>
      </svg>
    </div>
  )
}
