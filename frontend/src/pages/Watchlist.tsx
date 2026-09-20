/**
 * Person of Interest — enrol a photograph, then find where that person is.
 *
 * The governance is part of the interface, not a footnote: a legal basis is a
 * required field, a new subject is inert until a commander approves it, and
 * every sighting is a confidence that a human must confirm.
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Card,
  Empty,
  ErrorNote,
  Field,
  Icon,
  InfoNote,
  Loading,
  Modal,
  PageHeader,
  SeverityPill,
  StatusPill,
  Tabs,
  toast,
} from '../components/ui'
import { ApiError, api, type Paged, type WatchlistSubject } from '../lib/api'
import { useAuth } from '../lib/auth'
import { confidence, dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Options {
  categories: string[]
  priorities: string[]
  requires_approval: boolean
  face_backend: { backend: string; dim: number }
  body_backend: { backend: string; dim: number }
  augmentations: Record<string, string[]>
}

interface RecentMatch {
  id: string
  subject_id: string
  subject_label: string | null
  camera_id: string
  location_name: string | null
  matched_at: string
  modality: string
  confidence: number
  review_status: string
}

const TABS = [
  { key: 'all', label: 'All' },
  { key: 'active', label: 'Matching' },
  { key: 'pending', label: 'Awaiting approval' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function WatchlistPage() {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>('all')
  const [enrolOpen, setEnrolOpen] = useState(false)

  const params: Record<string, unknown> = { limit: 100 }
  if (tab !== 'all') params.status_filter = tab

  const { data, loading, error, refresh, reload } = useApi<Paged<WatchlistSubject>>(
    '/api/watchlist', params, { pollMs: 20000 },
  )
  const { data: options } = useApi<Options>('/api/watchlist/options')
  const { data: recent, refresh: refreshMatches } = useApi<{ items: RecentMatch[] }>(
    '/api/watchlist/matches/recent', { limit: 25 }, { pollMs: 15000 },
  )

  useLiveEvents(['finding', 'watchlist'], (event) => {
    if (event.topic === 'finding' && event.payload?.behavior !== 'watchlist_match') return
    refreshMatches()
    refresh()
  })

  useEffect(() => {
    document.title = 'Person of Interest · Eagles Eye'
  }, [])

  return (
    <>
      <PageHeader
        title="Person of Interest"
        subtitle="Enrolled references matched against every camera in scope"
        actions={
          can('watchlist:write') ? (
            <button className="btn-primary" onClick={() => setEnrolOpen(true)}>
              <Icon name="person_add" className="text-[16px]" />
              Enrol subject
            </button>
          ) : null
        }
      />

      <InfoNote icon="gavel">
        Enrolment requires a stated legal basis and a commander&rsquo;s approval before any matching
        starts. Sightings are appearance-based associations with a confidence — they are never an
        identification, and each one must be confirmed by a person.
        {options ? (
          <span className="block mt-1 mono text-outline">
            face: {options.face_backend.backend} · body: {options.body_backend.backend}
          </span>
        ) : null}
      </InfoNote>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" bodyClassName="p-0">
          <div className="px-space-md">
            <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />
          </div>

          {error ? (
            <div className="p-space-md"><ErrorNote message={error} onRetry={reload} /></div>
          ) : loading && !data ? (
            <Loading label="Loading subjects" />
          ) : !data || data.items.length === 0 ? (
            <Empty
              icon="person_search"
              title="No enrolled subjects"
              hint="Upload one or more photographs of the person you need to locate. Eagles Eye builds an augmented gallery so a change of hairstyle, glasses, lighting or camera angle does not break the match."
              action={
                can('watchlist:write') ? (
                  <button className="btn-primary" onClick={() => setEnrolOpen(true)}>
                    <Icon name="person_add" className="text-[16px]" />
                    Enrol subject
                  </button>
                ) : null
              }
            />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Subject</th>
                    <th>Category</th>
                    <th>Priority</th>
                    <th>Gallery</th>
                    <th>Sightings</th>
                    <th>Status</th>
                    <th>Expires</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((subject) => (
                    <tr key={subject.id}>
                      <td className="max-w-xs">
                        <Link to={`/watchlist/${subject.id}`} className="min-w-0 block group">
                          <span className="block truncate group-hover:text-primary-container">{subject.label}</span>
                          <span className="mono text-outline">{subject.case_reference ?? subject.id}</span>
                        </Link>
                      </td>
                      <td className="mono text-on-surface-variant">{titleCase(subject.category)}</td>
                      <td><SeverityPill severity={subject.priority === 'critical' ? 'critical' : subject.priority} /></td>
                      <td className="mono tabular-nums">
                        {subject.face_embedding_count}f / {subject.body_embedding_count}b
                      </td>
                      <td className="mono tabular-nums">{subject.total_sightings}</td>
                      <td>
                        <StatusPill
                          status={subject.matching_active ? 'online' : subject.expired ? 'error' : 'paused'}
                          label={subject.matching_active ? 'matching' : subject.expired ? 'expired' : subject.status}
                          pulse={subject.matching_active}
                        />
                      </td>
                      <td className="mono text-on-surface-variant whitespace-nowrap">
                        {subject.expires_at ? relativeTime(subject.expires_at).replace(' ago', '') : 'never'}
                      </td>
                      <td className="w-10">
                        <Link to={`/watchlist/${subject.id}`} className="btn-ghost btn-xs">
                          <Icon name="chevron_right" className="text-[16px]" />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Recent sightings" bodyClassName="p-0">
          {!recent || recent.items.length === 0 ? (
            <Empty icon="visibility_off" title="No sightings yet" hint="Approved subjects are matched on every running camera in scope." />
          ) : (
            <ul className="divide-y divide-outline-variant/25 max-h-[520px] overflow-y-auto">
              {recent.items.map((match) => (
                <li key={match.id} className="p-space-md">
                  <div className="flex items-center justify-between gap-2">
                    <Link
                      to={`/watchlist/${match.subject_id}`}
                      className="font-headline-sm text-headline-sm truncate hover:text-primary-container"
                    >
                      {match.subject_label ?? match.subject_id}
                    </Link>
                    <span className="mono font-semibold shrink-0">{confidence(match.confidence)}</span>
                  </div>
                  <p className="font-body-sm text-body-sm text-on-surface-variant truncate">
                    {match.location_name ?? match.camera_id}
                  </p>
                  <p className="mono text-outline">
                    {match.modality} · {relativeTime(match.matched_at)} ·{' '}
                    <span className={match.review_status === 'confirmed' ? 'text-emerald-700' : ''}>
                      {match.review_status}
                    </span>
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <EnrolDialog
        open={enrolOpen}
        options={options ?? null}
        onClose={() => setEnrolOpen(false)}
        onSaved={() => {
          setEnrolOpen(false)
          reload()
        }}
      />
    </>
  )
}

/* ----------------------------------------------------------------- enrol */
function EnrolDialog({
  open,
  options,
  onClose,
  onSaved,
}: {
  open: boolean
  options: Options | null
  onClose: () => void
  onSaved: () => void
}) {
  const [form, setForm] = useState({
    label: '',
    category: 'person_of_interest',
    priority: 'medium',
    description: '',
    legal_basis: '',
    case_reference: '',
    expires_in_days: 30,
  })
  const [files, setFiles] = useState<File[]>([])
  const [augment, setAugment] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ face_embeddings: number; body_embeddings: number; augmented_variants: number; warnings: string[] } | null>(null)

  useEffect(() => {
    if (open) {
      setForm({
        label: '', category: 'person_of_interest', priority: 'medium',
        description: '', legal_basis: '', case_reference: '', expires_in_days: 30,
      })
      setFiles([])
      setError(null)
      setResult(null)
    }
  }, [open])

  function set<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  async function submit() {
    setBusy(true)
    setError(null)
    try {
      const subject = await api.post<WatchlistSubject>('/api/watchlist', {
        ...form,
        description: form.description || null,
        case_reference: form.case_reference || null,
        expires_in_days: Number(form.expires_in_days) || null,
      })

      if (files.length) {
        const data = new FormData()
        files.forEach((file) => data.append('files', file))
        data.append('augment_images', augment ? 'true' : 'false')
        const upload = await api.upload<typeof result & object>(`/api/watchlist/${subject.id}/images`, data)
        setResult(upload as any)
      }

      toast(`${subject.label} enrolled — awaiting approval`)
      if (!files.length) onSaved()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Enrol a person of interest"
      width="max-w-2xl"
      footer={
        result ? (
          <button className="btn-primary" onClick={onSaved}>Done</button>
        ) : (
          <>
            <button className="btn-secondary" onClick={onClose}>Cancel</button>
            <button
              className="btn-primary"
              onClick={submit}
              disabled={busy || !form.label.trim() || form.legal_basis.trim().length < 8}
            >
              {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : null}
              Enrol
            </button>
          </>
        )
      }
    >
      {result ? (
        <div className="flex flex-col gap-space-md">
          <div className="flex items-center gap-2 text-emerald-800">
            <Icon name="check_circle" className="text-[20px]" />
            <span className="font-headline-md text-headline-md">Gallery built</span>
          </div>
          <dl className="grid grid-cols-3 gap-space-md">
            <Stat label="Face vectors" value={result.face_embeddings} />
            <Stat label="Body vectors" value={result.body_embeddings} />
            <Stat label="Augmented views" value={result.augmented_variants} />
          </dl>
          {result.warnings.map((warning) => (
            <p key={warning} className="font-body-sm text-body-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
              {warning}
            </p>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-space-lg">
          {error ? <ErrorNote message={error} /> : null}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
            <Field label="Subject label" required hint="A case label, not necessarily a real name.">
              <input className="input" value={form.label} onChange={(e) => set('label', e.target.value)} />
            </Field>
            <Field label="Case reference">
              <input className="input" value={form.case_reference} onChange={(e) => set('case_reference', e.target.value)} />
            </Field>
            <Field label="Category">
              <select className="input" value={form.category} onChange={(e) => set('category', e.target.value)}>
                {(options?.categories ?? ['person_of_interest']).map((c) => (
                  <option key={c} value={c}>{titleCase(c)}</option>
                ))}
              </select>
            </Field>
            <Field label="Priority">
              <select className="input" value={form.priority} onChange={(e) => set('priority', e.target.value)}>
                {(options?.priorities ?? ['medium']).map((p) => (
                  <option key={p} value={p}>{titleCase(p)}</option>
                ))}
              </select>
            </Field>
          </div>

          <Field
            label="Legal basis"
            required
            hint="State the authority, case or consent that permits this search. Recorded in the audit log and shown to the approving commander."
          >
            <textarea
              className="input h-20 py-2 resize-none"
              value={form.legal_basis}
              onChange={(e) => set('legal_basis', e.target.value)}
              placeholder="e.g. Missing-person report MP-2026-0142, filed by the duty officer."
            />
          </Field>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
            <Field label="Auto-expire after (days)" hint="Leave blank for no expiry.">
              <input
                type="number"
                className="input"
                value={form.expires_in_days}
                onChange={(e) => set('expires_in_days', Number(e.target.value))}
              />
            </Field>
            <Field label="Notes">
              <input className="input" value={form.description} onChange={(e) => set('description', e.target.value)} />
            </Field>
          </div>

          <Field
            label="Reference photographs"
            hint="Clear, front-facing images match best. Each one is augmented into many views — hairstyle, glasses, mask, lighting, blur, resolution and camera angle — so a single photo still matches in the field."
          >
            <input
              type="file"
              multiple
              accept="image/*"
              className="input h-auto py-1.5"
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            />
          </Field>

          {files.length > 0 ? (
            <div className="flex flex-wrap gap-space-sm">
              {files.map((file) => (
                <span key={file.name} className="pill bg-surface-container border-outline-variant text-on-surface">
                  <Icon name="image" className="text-[12px]" />
                  {file.name}
                </span>
              ))}
            </div>
          ) : null}

          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={augment} onChange={(e) => setAugment(e.target.checked)} />
            <span className="font-body-sm text-body-sm">
              Build the augmented gallery (recommended)
            </span>
          </label>
        </div>
      )}
    </Modal>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="font-headline-lg text-headline-lg">{value}</dd>
    </div>
  )
}
