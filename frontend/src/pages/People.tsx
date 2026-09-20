/** People & Tracking — live tracks, cross-camera subjects and trajectory replay. */
import { useEffect, useState } from 'react'
import { Card, Empty, ErrorNote, Icon, InfoNote, Kpi, Loading, PageHeader, Tabs, toast } from '../components/ui'
import { ApiError, api, type LiveTrack } from '../lib/api'
import { useAuth } from '../lib/auth'
import { confidence, dateTimeOf, duration, number, relativeTime } from '../lib/format'
import { useApi } from '../lib/hooks'

interface LiveCamera {
  camera_id: string
  camera_name: string
  tracks: LiveTrack[]
  measured_fps: number
}

interface Subject {
  global_id: string
  label: string
  cameras: string[]
  camera_count: number
  track_count: number
  views_held: number
  association_confidence: number
  first_seen: string | null
  last_seen: string | null
}

interface Trajectory {
  global_id: string
  legs: {
    track_id: string
    camera_id: string
    camera_name: string | null
    location: string | null
    first_seen: string
    last_seen: string
    dwell_seconds: number
    zones_visited: string[]
  }[]
  camera_sequence: string[]
  confidence_note: string
}

const TABS = [
  { key: 'live', label: 'Live tracks' },
  { key: 'subjects', label: 'Cross-camera subjects' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function PeoplePage() {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>('live')
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null)
  const [escalating, setEscalating] = useState<string | null>(null)

  const { data: live, loading, error, refresh } = useApi<{ cameras: LiveCamera[]; total_tracks: number }>(
    '/api/intel/tracks/live', undefined, { pollMs: 3000 },
  )
  const { data: subjects } = useApi<{ live: Subject[]; stored: unknown[]; note: string; embedding_backend: { backend: string } }>(
    '/api/intel/subjects', { limit: 100 }, { pollMs: 10000 },
  )

  useEffect(() => {
    document.title = 'People & Tracking · Eagles Eye'
  }, [])

  async function showTrajectory(globalId: string) {
    try {
      setTrajectory(await api.get<Trajectory>(`/api/intel/subjects/${globalId}/trajectory`))
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function escalate(globalId: string) {
    const justification = prompt(
      'Identity escalation request\n\nSubjects are anonymous by default. Releasing identity-bearing data requires a commander’s approval.\n\nState the lawful basis and case reference:',
    )
    if (!justification || justification.trim().length < 15) {
      if (justification !== null) toast('A substantive justification is required', 'error')
      return
    }
    setEscalating(globalId)
    try {
      await api.post('/api/intel/identity/escalate', { global_id: globalId, justification })
      toast('Escalation request sent to a commander for approval')
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setEscalating(null)
    }
  }

  const allTracks = (live?.cameras ?? []).flatMap((camera) =>
    camera.tracks.map((track) => ({ ...track, camera_id: camera.camera_id, camera_name: camera.camera_name })),
  )
  const people = allTracks.filter((t) => t.class_name === 'person')

  return (
    <>
      <PageHeader
        title="People & Tracking"
        subtitle="Anonymous tracks, cross-camera association and trajectory reconstruction"
      />

      <InfoNote icon="privacy_tip">
        Everyone here is an anonymous track. A cross-camera match is an appearance-based association
        with a stated confidence — never a claim that two observations are definitely the same person.
        {subjects ? (
          <span className="block mt-1 mono text-outline">
            embedding backend: {subjects.embedding_backend.backend}
          </span>
        ) : null}
      </InfoNote>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi label="Live tracks" value={number(live?.total_tracks ?? 0)} footer={`${live?.cameras.length ?? 0} cameras`} />
        <Kpi label="People in view" value={number(people.length)} />
        <Kpi label="Cross-camera subjects" value={number(subjects?.live.length ?? 0)} />
        <Kpi
          label="Seen on 2+ cameras"
          value={number((subjects?.live ?? []).filter((s) => s.camera_count > 1).length)}
          footer="association candidates"
        />
      </section>

      {error ? <ErrorNote message={error} onRetry={refresh} /> : null}

      <Card bodyClassName="p-0">
        <div className="px-space-md">
          <Tabs
            tabs={[
              { key: 'live' as const, label: 'Live tracks', count: allTracks.length },
              { key: 'subjects' as const, label: 'Cross-camera subjects', count: subjects?.live.length ?? 0 },
            ]}
            active={tab}
            onChange={setTab}
          />
        </div>

        {tab === 'live' ? (
          loading && !live ? (
            <Loading />
          ) : allTracks.length === 0 ? (
            <Empty icon="person_off" title="No active tracks" hint="Start a camera to begin tracking." />
          ) : (
            <div className="table-wrap max-h-[60vh] overflow-y-auto">
              <table className="tbl">
                <thead className="sticky top-0 z-10">
                  <tr>
                    <th>Track</th>
                    <th>Class</th>
                    <th>Camera</th>
                    <th>Subject</th>
                    <th>Speed</th>
                    <th>Heading</th>
                    <th>Zone</th>
                    <th>Face</th>
                    <th>Observed</th>
                  </tr>
                </thead>
                <tbody>
                  {allTracks.map((track) => (
                    <tr key={`${track.camera_id}-${track.track_id}`}>
                      <td className="mono font-semibold">#{track.track_id}</td>
                      <td>{track.class_name}</td>
                      <td className="truncate max-w-[10rem]">{track.camera_name}</td>
                      <td className="mono text-on-surface-variant">
                        {track.global_id ? (
                          <button className="hover:text-primary-container" onClick={() => showTrajectory(track.global_id!)}>
                            {track.global_id.slice(0, 12)}
                          </button>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="mono tabular-nums">{track.speed.toFixed(0)} px/s</td>
                      <td className="mono tabular-nums">
                        {track.heading_deg === null ? '—' : `${track.heading_deg.toFixed(0)}°`}
                      </td>
                      <td className="mono text-on-surface-variant truncate max-w-[8rem]">{track.zone_id ?? '—'}</td>
                      <td className="mono text-on-surface-variant">{track.face_visibility}</td>
                      <td className="mono">{duration(track.duration_seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : !subjects || subjects.live.length === 0 ? (
          <Empty icon="link_off" title="No cross-camera subjects yet" hint="Subjects appear once the Re-ID agent has enough views to associate." />
        ) : (
          <div className="table-wrap max-h-[60vh] overflow-y-auto">
            <table className="tbl">
              <thead className="sticky top-0 z-10">
                <tr>
                  <th>Subject</th>
                  <th>Cameras</th>
                  <th>Tracks</th>
                  <th>Views held</th>
                  <th>Association confidence</th>
                  <th>First seen</th>
                  <th>Last seen</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {subjects.live.map((subject) => (
                  <tr key={subject.global_id}>
                    <td className="mono font-semibold">{subject.label}</td>
                    <td className="mono">{subject.camera_count}</td>
                    <td className="mono">{subject.track_count}</td>
                    <td className="mono">{subject.views_held}</td>
                    <td className="mono tabular-nums">{confidence(subject.association_confidence)}</td>
                    <td className="mono" title={dateTimeOf(subject.first_seen)}>{relativeTime(subject.first_seen)}</td>
                    <td className="mono">{relativeTime(subject.last_seen)}</td>
                    <td className="w-56">
                      <div className="flex gap-1">
                        <button className="btn-secondary btn-xs" onClick={() => showTrajectory(subject.global_id)}>
                          Trajectory
                        </button>
                        {can('identity:escalate') ? (
                          <button
                            className="btn-ghost btn-xs"
                            onClick={() => escalate(subject.global_id)}
                            disabled={escalating === subject.global_id}
                            title="Requires commander approval"
                          >
                            Escalate
                          </button>
                        ) : null}
                      </div>
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
          title={`Trajectory · ${trajectory.global_id}`}
          actions={
            <button className="btn-ghost btn-xs" onClick={() => setTrajectory(null)}>
              <Icon name="close" className="text-[16px]" />
            </button>
          }
        >
          {trajectory.legs.length === 0 ? (
            <Empty icon="timeline" title="No recorded legs for this subject" />
          ) : (
            <div className="flex flex-col gap-space-md">
              <ol className="flex items-center gap-1 flex-wrap">
                {trajectory.legs.map((leg, index) => (
                  <li key={leg.track_id} className="flex items-center gap-1">
                    {index > 0 ? <Icon name="arrow_forward" className="text-[14px] text-outline" /> : null}
                    <span className="pill bg-surface-container border-outline-variant text-on-surface">
                      {leg.camera_name ?? leg.camera_id}
                    </span>
                  </li>
                ))}
              </ol>
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Camera</th>
                      <th>Location</th>
                      <th>Entered</th>
                      <th>Left</th>
                      <th>Dwell</th>
                      <th>Zones</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trajectory.legs.map((leg) => (
                      <tr key={leg.track_id}>
                        <td>{leg.camera_name ?? leg.camera_id}</td>
                        <td className="text-on-surface-variant">{leg.location ?? '—'}</td>
                        <td className="mono">{dateTimeOf(leg.first_seen)}</td>
                        <td className="mono">{dateTimeOf(leg.last_seen)}</td>
                        <td className="mono">{duration(leg.dwell_seconds)}</td>
                        <td className="mono text-on-surface-variant">{leg.zones_visited.join(', ') || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant">{trajectory.confidence_note}</p>
            </div>
          )}
        </Card>
      ) : null}
    </>
  )
}
