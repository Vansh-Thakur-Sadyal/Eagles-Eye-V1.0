/**
 * Incident workspace: narrative, explainable risk breakdown, timeline and the
 * recommended-action queue. High-consequence actions are visibly gated.
 */
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Card,
  Empty,
  ErrorNote,
  Icon,
  InfoNote,
  Loading,
  Meter,
  PageHeader,
  SeverityPill,
  toast,
} from '../components/ui'
import { ApiError, api, streamUrl, type Camera, type Incident, type RecommendedAction, type TimelineEntry } from '../lib/api'
import { useAuth } from '../lib/auth'
import { behaviorIcon, behaviorLabel, confidence, dateTimeOf, duration, relativeTime, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Detail extends Incident {
  timeline: TimelineEntry[]
  camera: Camera | null
  behavior_events: {
    id: string
    behavior: string
    confidence: number
    severity: string
    started_at: string
    duration_seconds: number
    explanation: string | null
    agent: string
  }[]
}

const STATUSES = ['open', 'acknowledged', 'dispatched', 'investigating', 'resolved', 'false_positive']

export default function IncidentDetailPage() {
  const { incidentId } = useParams<{ incidentId: string }>()
  const { can } = useAuth()
  const { data, loading, error, refresh, reload } = useApi<Detail>(
    incidentId ? `/api/incidents/${incidentId}` : null, undefined, { pollMs: 15000 },
  )
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  useLiveEvents(['incident'], (event) => {
    if (event.payload?.id === incidentId || event.payload?.incident_id === incidentId) refresh()
  })

  useEffect(() => {
    document.title = data ? `${data.title} · Eagles Eye` : 'Incident · Eagles Eye'
  }, [data])

  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (loading && !data) return <Loading label="Loading incident" />
  if (!data) return <Empty icon="search_off" title="Incident not found" />

  async function setStatus(status: string) {
    setBusy(status)
    try {
      await api.post(`/api/incidents/${incidentId}/status`, { status, note: note || undefined })
      toast(`Status set to ${titleCase(status)}`)
      setNote('')
      refresh()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setBusy(null)
    }
  }

  async function runAction(action: RecommendedAction) {
    const justification = action.requires_human_approval
      ? prompt(`${action.label}\n\nThis has a real-world consequence (${action.consequence}).\nState your justification:`)
      : ''
    if (action.requires_human_approval && !justification) return

    setBusy(action.key)
    try {
      const result = await api.post<{ status: string; requires_human_approval: boolean }>(
        `/api/incidents/${incidentId}/actions`,
        { action_key: action.key, justification: justification || undefined },
      )
      toast(
        result.status === 'pending_approval'
          ? `${action.label} sent to a commander for approval`
          : `${action.label} approved`,
      )
      refresh()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setBusy(null)
    }
  }

  async function addNote() {
    if (!note.trim()) return
    try {
      await api.post(`/api/incidents/${incidentId}/notes`, { text: note.trim() })
      setNote('')
      refresh()
      toast('Note added')
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const counted = data.risk_factors.filter((f) => f.counted)
  const observability = data.risk_factors.filter((f) => !f.counted)
  const maxContribution = Math.max(1, ...counted.map((f) => f.contribution))
  const canWrite = can('incident:write')

  return (
    <>
      <PageHeader
        title={data.title}
        subtitle={
          <span className="flex items-center gap-2 flex-wrap">
            <span className="mono">{data.id}</span>
            <span>·</span>
            <span>{data.camera?.name ?? data.camera_id ?? 'unknown camera'}</span>
            <span>·</span>
            <span>started {dateTimeOf(data.started_at)}</span>
          </span>
        }
        actions={
          <>
            <Link to="/incidents" className="btn-secondary">
              <Icon name="arrow_back" className="text-[16px]" />
              All incidents
            </Link>
            <SeverityPill severity={data.severity} />
          </>
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        {/* ------------------------------------------------------- narrative */}
        <div className="xl:col-span-2 flex flex-col gap-space-lg">
          <Card title="Incident Commander assessment">
            <div className="flex flex-col gap-space-md">
              <div className="flex items-center gap-space-lg flex-wrap">
                <div className="flex items-baseline gap-2">
                  <span className="font-display-lg text-display-lg font-bold">{data.risk_score.toFixed(0)}</span>
                  <span className="mono text-on-surface-variant">/ 100 risk</span>
                </div>
                <div className="flex-1 min-w-[12rem]">
                  <Meter
                    value={data.risk_score}
                    tone={data.severity === 'critical' ? 'bg-red-600' : data.severity === 'high' ? 'bg-amber-500' : 'bg-primary-container'}
                  />
                </div>
                <span className="mono text-on-surface-variant">
                  confidence {confidence(data.confidence)}
                </span>
              </div>

              <p className="font-body-lg text-body-lg text-on-surface">{data.summary}</p>

              <InfoNote icon="psychology">
                {data.explanation}
                <span className="block mt-1 text-outline">
                  Narration source: {String((data.meta as any)?.narration_source ?? 'template')}
                </span>
              </InfoNote>
            </div>
          </Card>

          <Card title="Why this score">
            {counted.length === 0 ? (
              <Empty icon="functions" title="No weighted signals are active" />
            ) : (
              <div className="flex flex-col gap-space-md">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Signal</th>
                      <th className="w-16">Weight</th>
                      <th className="w-20">Confidence</th>
                      <th className="w-20">Recency</th>
                      <th className="w-28">Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {counted.map((factor) => (
                      <tr key={factor.behavior}>
                        <td>
                          <span className="flex items-center gap-2">
                            <Icon name={behaviorIcon(factor.behavior)} className="text-[16px] text-on-surface-variant" />
                            {behaviorLabel(factor.behavior)}
                          </span>
                        </td>
                        <td className="mono tabular-nums">{factor.weight}</td>
                        <td className="mono tabular-nums">{confidence(factor.confidence)}</td>
                        <td className="mono tabular-nums">{factor.recency_factor.toFixed(2)}</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span className="mono tabular-nums w-10">{factor.contribution.toFixed(1)}</span>
                            <Meter value={factor.contribution} max={maxContribution} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {observability.length > 0 ? (
                  <div className="rounded-lg bg-surface-container-low border border-outline-variant/40 p-space-sm">
                    <p className="font-headline-sm text-headline-sm text-on-surface-variant mb-1">
                      Observability signals — deliberately weighted 0
                    </p>
                    {observability.map((factor) => (
                      <p key={factor.behavior} className="font-body-sm text-body-sm text-on-surface-variant">
                        {behaviorLabel(factor.behavior)}: {factor.note ?? 'carries no risk weight'}
                      </p>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
          </Card>

          <Card title={`Timeline · ${data.timeline.length} entries`} bodyClassName="p-0">
            <ol className="divide-y divide-outline-variant/25">
              {data.timeline.map((entry) => (
                <li key={entry.id} className="p-space-md flex gap-space-md">
                  <div className="flex flex-col items-center shrink-0">
                    <span className={`w-6 h-6 rounded-full flex items-center justify-center ${kindTone(entry.kind)}`}>
                      <Icon name={kindIcon(entry.kind)} className="text-[14px]" />
                    </span>
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-headline-sm text-headline-sm">{titleCase(entry.kind)}</span>
                      <span className="mono text-on-surface-variant shrink-0" title={dateTimeOf(entry.at)}>
                        {dateTimeOf(entry.at).slice(-8)}
                      </span>
                    </div>
                    <p className="font-body-sm text-body-sm text-on-surface-variant">{entry.text}</p>
                    <p className="mono text-outline mt-0.5">
                      {entry.actor}
                      {entry.confidence !== null ? ` · confidence ${confidence(entry.confidence)}` : ''}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          </Card>

          {data.behavior_events.length > 0 ? (
            <Card title="Contributing observations" bodyClassName="p-0">
              <div className="table-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Behaviour</th>
                      <th>Agent</th>
                      <th>Confidence</th>
                      <th>Duration</th>
                      <th>When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.behavior_events.map((event) => (
                      <tr key={event.id}>
                        <td title={event.explanation ?? ''}>{behaviorLabel(event.behavior)}</td>
                        <td className="mono text-on-surface-variant">{event.agent}</td>
                        <td className="mono tabular-nums">{confidence(event.confidence)}</td>
                        <td className="mono">{duration(event.duration_seconds)}</td>
                        <td className="mono text-on-surface-variant">{relativeTime(event.started_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ) : null}
        </div>

        {/* --------------------------------------------------------- actions */}
        <div className="flex flex-col gap-space-lg">
          {data.camera?.running ? (
            <Card title="Live view" bodyClassName="p-0">
              <img
                src={streamUrl(data.camera.id)}
                alt="Incident camera live view"
                className="w-full aspect-video object-cover rounded-b-xl"
              />
            </Card>
          ) : null}

          <Card title="Recommended actions">
            <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
              Eagles Eye recommends; an operator decides. Anything with a real-world consequence needs a
              commander&rsquo;s approval.
            </p>
            <ul className="flex flex-col gap-space-sm">
              {data.recommended_actions.map((action) => (
                <li
                  key={action.key}
                  className="flex items-start justify-between gap-space-md p-space-sm rounded-lg border border-outline-variant/50 bg-surface"
                >
                  <div className="min-w-0">
                    <p className="font-headline-sm text-headline-sm">{action.label}</p>
                    <p className="mono text-on-surface-variant">
                      {action.consequence}
                      {action.requires_human_approval ? ' · approval required' : ''}
                    </p>
                    {action.status !== 'proposed' ? (
                      <p className="mono text-emerald-700 mt-0.5">{titleCase(action.status)}</p>
                    ) : null}
                  </div>
                  {canWrite && action.status === 'proposed' ? (
                    <button
                      className={action.requires_human_approval ? 'btn-secondary btn-xs' : 'btn-primary btn-xs'}
                      onClick={() => runAction(action)}
                      disabled={busy === action.key}
                    >
                      {action.requires_human_approval ? 'Request' : 'Approve'}
                    </button>
                  ) : null}
                </li>
              ))}
            </ul>
          </Card>

          {canWrite ? (
            <Card title="Disposition">
              <div className="flex flex-col gap-space-md">
                <label className="block">
                  <span className="label">Note</span>
                  <textarea
                    className="input h-20 py-2 resize-none"
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Optional — attached to the status change or added on its own."
                  />
                </label>
                <button className="btn-secondary" onClick={addNote} disabled={!note.trim()}>
                  <Icon name="add_comment" className="text-[16px]" />
                  Add note only
                </button>
                <div className="grid grid-cols-2 gap-space-sm">
                  {STATUSES.filter((s) => s !== data.status).map((status) => (
                    <button
                      key={status}
                      className={status === 'resolved' ? 'btn-primary' : 'btn-secondary'}
                      onClick={() => setStatus(status)}
                      disabled={busy === status}
                    >
                      {titleCase(status)}
                    </button>
                  ))}
                </div>
              </div>
            </Card>
          ) : null}

          <Card title="Context">
            <dl className="grid grid-cols-2 gap-space-md">
              <Detail label="Status" value={titleCase(data.status)} />
              <Detail label="Event type" value={behaviorLabel(data.event_type)} />
              <Detail label="Zone" value={String((data.location as any)?.zone_name ?? data.zone_id ?? '—')} />
              <Detail label="Floor" value={String((data.location as any)?.floor ?? '—')} />
              <Detail label="Tracks" value={data.track_ids.join(', ') || '—'} />
              <Detail label="Objects" value={data.object_ids.join(', ') || '—'} />
              <Detail label="Last update" value={relativeTime(data.last_update_at)} />
              <Detail label="Assigned" value={data.assigned_to ?? 'unassigned'} />
            </dl>
          </Card>
        </div>
      </div>
    </>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface truncate" title={value}>{value}</dd>
    </div>
  )
}

function kindIcon(kind: string): string {
  return (
    { detection: 'sensors', assessment: 'analytics', notification: 'campaign', action: 'task_alt', operator: 'person', observation: 'visibility' }[
      kind
    ] ?? 'circle'
  )
}

function kindTone(kind: string): string {
  return (
    {
      detection: 'bg-blue-50 text-blue-700',
      assessment: 'bg-violet-50 text-violet-700',
      notification: 'bg-amber-50 text-amber-700',
      action: 'bg-emerald-50 text-emerald-700',
      operator: 'bg-surface-container-high text-on-surface',
    }[kind] ?? 'bg-surface-container text-on-surface-variant'
  )
}
