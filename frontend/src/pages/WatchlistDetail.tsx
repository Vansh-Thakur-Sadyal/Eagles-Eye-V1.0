/** One subject: gallery, approval, live location answer and sighting review. */
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  Card,
  Empty,
  ErrorNote,
  Icon,
  InfoNote,
  Loading,
  PageHeader,
  SeverityPill,
  StatusPill,
  toast,
} from '../components/ui'
import { ApiError, api, type Paged, type WatchlistSubject } from '../lib/api'
import { useAuth } from '../lib/auth'
import { confidence, dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Locate {
  subject_id: string
  label: string
  located: boolean
  statement: string
  matching_active?: boolean
  latest?: {
    camera_id: string
    camera_name: string | null
    location: string | null
    zone_id: string | null
    floor: number | null
    latitude: number | null
    longitude: number | null
    matched_at: string
    confidence: number
    modality: string
    review_status: string
  }
  movement_path?: {
    camera_id: string
    camera_name: string | null
    location: string | null
    matched_at: string
    confidence: number
    modality: string
    review_status: string
  }[]
  cameras_seen?: string[]
  requires_human_confirmation?: boolean
}

interface Sighting {
  id: string
  camera_id: string
  camera_name: string | null
  camera_location: string | null
  location_name: string | null
  matched_at: string
  modality: string
  confidence: number
  similarity: number
  face_visibility: string
  review_status: string
  reviewed_by: string | null
  latitude: number | null
  longitude: number | null
  floor: number | null
}

export default function WatchlistDetailPage() {
  const { subjectId } = useParams<{ subjectId: string }>()
  const { can } = useAuth()
  const navigate = useNavigate()
  const [windowMinutes, setWindowMinutes] = useState(120)

  const { data: subject, loading, error, refresh } = useApi<WatchlistSubject>(
    subjectId ? `/api/watchlist/${subjectId}` : null, undefined, { pollMs: 20000 },
  )
  const { data: locate, refresh: refreshLocate } = useApi<Locate>(
    subjectId ? `/api/watchlist/${subjectId}/locate` : null,
    { within_minutes: windowMinutes, min_confidence: 0.6 },
    { pollMs: 15000 },
  )
  const { data: sightings, refresh: refreshSightings } = useApi<Paged<Sighting>>(
    subjectId ? `/api/watchlist/${subjectId}/sightings` : null, { limit: 100 }, { pollMs: 20000 },
  )

  useLiveEvents(['finding'], (event) => {
    if (event.payload?.behavior === 'watchlist_match' && event.payload?.evidence?.subject_id === subjectId) {
      refreshLocate()
      refreshSightings()
      refresh()
    }
  })

  useEffect(() => {
    document.title = subject ? `${subject.label} · Eagles Eye` : 'Subject · Eagles Eye'
  }, [subject])

  if (error) return <ErrorNote message={error} onRetry={refresh} />
  if (loading && !subject) return <Loading label="Loading subject" />
  if (!subject) return <Empty icon="person_off" title="Subject not found" />

  async function decide(decision: 'approve' | 'reject') {
    const note = prompt(
      decision === 'approve'
        ? 'Approve this enrolment? Matching will begin on every camera in scope.\n\nOptional note:'
        : 'Reject this enrolment?\n\nOptional note:',
    )
    if (note === null) return
    try {
      await api.post(`/api/watchlist/${subjectId}/approve`, { decision, note: note || undefined })
      toast(decision === 'approve' ? 'Matching started' : 'Enrolment rejected')
      refresh()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function setStatus(status: string) {
    try {
      await api.patch(`/api/watchlist/${subjectId}`, { status })
      toast(`Subject ${status}`)
      refresh()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function remove() {
    if (!confirm(`Delete "${subject!.label}" and stop all matching? This also removes the stored reference images.`)) return
    try {
      await api.delete(`/api/watchlist/${subjectId}`, { purge_files: true })
      toast('Subject removed')
      navigate('/watchlist')
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function review(sighting: Sighting, status: 'confirmed' | 'rejected' | 'uncertain') {
    try {
      await api.post(`/api/watchlist/matches/${sighting.id}/review`, { review_status: status })
      toast(`Sighting marked ${status}`)
      refreshSightings()
      refreshLocate()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const references = subject.images.filter((i) => i.kind === 'reference')
  const augmented = subject.images.filter((i) => i.kind === 'augmented')

  return (
    <>
      <PageHeader
        title={subject.label}
        subtitle={
          <span className="flex items-center gap-2 flex-wrap">
            <span className="mono">{subject.case_reference ?? subject.id}</span>
            <span>·</span>
            <span>{titleCase(subject.category)}</span>
            <span>·</span>
            <span>enrolled by {subject.requested_by ?? 'unknown'}</span>
          </span>
        }
        actions={
          <>
            <Link to="/watchlist" className="btn-secondary">
              <Icon name="arrow_back" className="text-[16px]" />
              All subjects
            </Link>
            <SeverityPill severity={subject.priority === 'critical' ? 'critical' : subject.priority} />
            <StatusPill
              status={subject.matching_active ? 'online' : subject.expired ? 'error' : 'paused'}
              label={subject.matching_active ? 'matching' : subject.expired ? 'expired' : subject.status}
              pulse={subject.matching_active}
            />
          </>
        }
      />

      {subject.status === 'pending' ? (
        <div className="flex items-start gap-space-md p-space-md rounded-lg bg-amber-50 border border-amber-300">
          <Icon name="gavel" className="text-[20px] text-amber-700 shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <p className="font-headline-sm text-headline-sm text-amber-900">Awaiting commander approval</p>
            <p className="font-body-sm text-body-sm text-amber-800">
              No camera is matching this subject yet. Stated legal basis: “{subject.legal_basis}”
            </p>
          </div>
          {can('watchlist:approve') ? (
            <div className="flex gap-space-sm shrink-0">
              <button className="btn-secondary btn-xs" onClick={() => decide('reject')}>Reject</button>
              <button className="btn-primary btn-xs" onClick={() => decide('approve')}>Approve</button>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        {/* ------------------------------------------------------- location */}
        <Card
          className="xl:col-span-2"
          title="Where is this person?"
          actions={
            <select
              className="input w-32 h-6 text-label-sm"
              value={windowMinutes}
              onChange={(e) => setWindowMinutes(Number(e.target.value))}
            >
              {[15, 60, 120, 480, 1440].map((m) => (
                <option key={m} value={m}>last {m < 60 ? `${m} min` : `${m / 60} h`}</option>
              ))}
            </select>
          }
        >
          {!locate ? (
            <Loading />
          ) : (
            <div className="flex flex-col gap-space-md">
              <div
                className={`flex items-start gap-space-md p-space-md rounded-lg border ${
                  locate.located
                    ? 'bg-emerald-50 border-emerald-200'
                    : 'bg-surface-container-low border-outline-variant/40'
                }`}
              >
                <Icon
                  name={locate.located ? 'my_location' : 'location_off'}
                  className={`text-[22px] shrink-0 mt-0.5 ${locate.located ? 'text-emerald-700' : 'text-outline'}`}
                />
                <p className="font-body-lg text-body-lg text-on-surface">{locate.statement}</p>
              </div>

              {locate.located && locate.latest ? (
                <dl className="grid grid-cols-2 sm:grid-cols-4 gap-space-md">
                  <Detail label="Camera" value={locate.latest.camera_name ?? locate.latest.camera_id} />
                  <Detail label="Location" value={locate.latest.location ?? '—'} />
                  <Detail label="Confidence" value={confidence(locate.latest.confidence)} />
                  <Detail label="Modality" value={locate.latest.modality} />
                  <Detail label="Floor" value={String(locate.latest.floor ?? '—')} />
                  <Detail
                    label="Coordinates"
                    value={
                      locate.latest.latitude != null
                        ? `${locate.latest.latitude.toFixed(5)}, ${locate.latest.longitude?.toFixed(5)}`
                        : 'not geolocated'
                    }
                  />
                  <Detail label="Seen" value={relativeTime(locate.latest.matched_at)} />
                  <Detail label="Review" value={titleCase(locate.latest.review_status)} />
                </dl>
              ) : null}

              {locate.movement_path && locate.movement_path.length > 1 ? (
                <div>
                  <p className="label">Movement across cameras</p>
                  <ol className="flex items-center gap-1 flex-wrap">
                    {locate.movement_path
                      .slice()
                      .reverse()
                      .map((step, index) => (
                        <li key={`${step.camera_id}-${step.matched_at}`} className="flex items-center gap-1">
                          {index > 0 ? <Icon name="arrow_forward" className="text-[14px] text-outline" /> : null}
                          <span
                            className="pill bg-surface-container border-outline-variant text-on-surface"
                            title={`${dateTimeOf(step.matched_at)} · ${confidence(step.confidence)}`}
                          >
                            {step.camera_name ?? step.camera_id}
                          </span>
                        </li>
                      ))}
                  </ol>
                </div>
              ) : null}

              {locate.located ? (
                <InfoNote icon="how_to_reg">
                  This is an appearance-based association, not an identification. Confirm or reject
                  each sighting below before acting on it.
                </InfoNote>
              ) : null}
            </div>
          )}
        </Card>

        {/* -------------------------------------------------------- gallery */}
        <Card title={`Match gallery · ${subject.face_embedding_count + subject.body_embedding_count} vectors`}>
          <div className="flex flex-col gap-space-md">
            <dl className="grid grid-cols-3 gap-space-md">
              <Detail label="References" value={String(references.length)} />
              <Detail label="Augmented" value={String(augmented.length)} />
              <Detail label="Sightings" value={String(subject.total_sightings)} />
            </dl>

            {references.length === 0 ? (
              <Empty icon="add_photo_alternate" title="No reference images" hint="Matching cannot run without at least one photograph." />
            ) : (
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                {subject.face_embedding_count} face vector(s) and {subject.body_embedding_count} body
                vector(s) are loaded. Body-only matches are capped well below face matches, because
                clothing is not an identity.
              </p>
            )}

            {augmented.length > 0 ? (
              <details className="rounded-lg border border-outline-variant/50">
                <summary className="px-space-sm h-8 flex items-center cursor-pointer font-headline-sm text-headline-sm text-on-surface-variant">
                  Augmentations applied
                </summary>
                <div className="p-space-sm flex flex-wrap gap-1 border-t border-outline-variant/40 max-h-40 overflow-y-auto">
                  {Array.from(new Set(augmented.map((a) => a.augmentation).filter(Boolean))).map((aug) => (
                    <span key={aug} className="pill bg-surface-container border-outline-variant text-on-surface-variant">
                      {aug}
                    </span>
                  ))}
                </div>
              </details>
            ) : null}

            <dl className="grid grid-cols-2 gap-space-md border-t border-outline-variant/40 pt-space-md">
              <Detail label="Approved by" value={subject.approved_by ?? '—'} />
              <Detail label="Approved at" value={subject.approved_at ? dateTimeOf(subject.approved_at) : '—'} />
              <Detail label="Expires" value={subject.expires_at ? dateTimeOf(subject.expires_at) : 'never'} />
              <Detail label="Alert on match" value={subject.alert_on_match ? 'yes' : 'no'} />
            </dl>

            {can('watchlist:write') ? (
              <div className="flex flex-wrap gap-space-sm border-t border-outline-variant/40 pt-space-md">
                {subject.status === 'active' ? (
                  <button className="btn-secondary btn-xs" onClick={() => setStatus('paused')}>Pause matching</button>
                ) : subject.status === 'paused' && can('watchlist:approve') ? (
                  <button className="btn-primary btn-xs" onClick={() => setStatus('active')}>Resume matching</button>
                ) : null}
                {can('watchlist:approve') ? (
                  <button className="btn-danger btn-xs" onClick={remove}>Delete subject</button>
                ) : null}
              </div>
            ) : null}
          </div>
        </Card>
      </div>

      {/* -------------------------------------------------------- sightings */}
      <Card title={`Sightings · ${sightings?.total ?? 0}`} bodyClassName="p-0">
        {!sightings || sightings.items.length === 0 ? (
          <Empty
            icon="visibility_off"
            title="No sightings recorded"
            hint={subject.matching_active ? 'Matching is active across every camera in scope.' : 'Matching has not started for this subject.'}
          />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Camera</th>
                  <th>Location</th>
                  <th>Modality</th>
                  <th>Similarity</th>
                  <th>Confidence</th>
                  <th>Face</th>
                  <th>Review</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sightings.items.map((sighting) => (
                  <tr key={sighting.id}>
                    <td className="mono whitespace-nowrap" title={dateTimeOf(sighting.matched_at)}>
                      {relativeTime(sighting.matched_at)}
                    </td>
                    <td className="truncate max-w-[10rem]">{sighting.camera_name ?? sighting.camera_id}</td>
                    <td className="truncate max-w-[12rem] text-on-surface-variant">
                      {sighting.location_name ?? sighting.camera_location ?? '—'}
                    </td>
                    <td className="mono">{sighting.modality}</td>
                    <td className="mono tabular-nums">{sighting.similarity.toFixed(3)}</td>
                    <td className="mono tabular-nums font-semibold">{confidence(sighting.confidence)}</td>
                    <td className="mono text-on-surface-variant">{sighting.face_visibility}</td>
                    <td>
                      <span
                        className={`mono ${
                          sighting.review_status === 'confirmed'
                            ? 'text-emerald-700'
                            : sighting.review_status === 'rejected'
                              ? 'text-red-700'
                              : 'text-on-surface-variant'
                        }`}
                      >
                        {sighting.review_status}
                        {sighting.reviewed_by ? ` · ${sighting.reviewed_by}` : ''}
                      </span>
                    </td>
                    <td className="w-40">
                      {sighting.review_status === 'unreviewed' ? (
                        <div className="flex gap-1">
                          <button className="btn-secondary btn-xs" onClick={() => review(sighting, 'confirmed')}>
                            Confirm
                          </button>
                          <button className="btn-ghost btn-xs" onClick={() => review(sighting, 'rejected')}>
                            Reject
                          </button>
                        </div>
                      ) : null}
                    </td>
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

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface truncate" title={value}>{value}</dd>
    </div>
  )
}
