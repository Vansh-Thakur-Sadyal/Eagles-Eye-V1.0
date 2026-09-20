/**
 * Settings — every threshold, weight, model and automation binding the platform
 * uses, edited live. Nothing here is a mock: each control writes to the policy
 * file or the database and the orchestrator reloads without a restart.
 */
import { useEffect, useMemo, useState } from 'react'
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
  Tabs,
  Toggle,
  toast,
} from '../components/ui'
import { ApiError, api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { behaviorLabel, dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi, useGpu } from '../lib/hooks'

type Policy = Record<string, Record<string, unknown>>

const SECTION_HELP: Record<string, string> = {
  threat_weights: 'How much each signal contributes to the risk score. Observability signals are kept at 0 on purpose.',
  threat_bands: 'Score thresholds for each severity band.',
  behavior: 'Thresholds for loitering, running, counter-flow and sudden movement.',
  following: 'How persistent a following pattern must be before it is raised for review.',
  crowd: 'Density bands, flow window and surge threshold.',
  object: 'Which object classes are watched, and how long before an object counts as unattended.',
  reid: 'Cross-camera association threshold and gallery limits.',
  watchlist: 'Match thresholds and whether enrolment requires approval.',
  occlusion: 'Minimum observable face size before identity confidence drops.',
  privacy: 'Redaction, escalation gating and retention.',
}

const TABS = [
  { key: 'thresholds', label: 'Detection policy' },
  { key: 'scoring', label: 'Risk scoring' },
  { key: 'compute', label: 'Compute & models' },
  { key: 'workflows', label: 'Automation' },
  { key: 'users', label: 'Operators' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function SettingsPage({ initialTab }: { initialTab?: TabKey }) {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>(initialTab ?? 'thresholds')

  useEffect(() => {
    document.title = 'Settings · Eagles Eye'
  }, [])

  return (
    <>
      <PageHeader title="Settings" subtitle="Live configuration — changes apply without a restart" />
      <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />

      {tab === 'thresholds' ? <PolicyEditor sections={['behavior', 'following', 'crowd', 'object', 'reid', 'occlusion', 'watchlist', 'privacy']} /> : null}
      {tab === 'scoring' ? <PolicyEditor sections={['threat_weights', 'threat_bands']} /> : null}
      {tab === 'compute' ? <ComputePanel /> : null}
      {tab === 'workflows' ? <WorkflowsPanel /> : null}
      {tab === 'users' ? <UsersPanel /> : null}
    </>
  )
}

/* ------------------------------------------------------------ policy edit */
function PolicyEditor({ sections }: { sections: string[] }) {
  const { can } = useAuth()
  const editable = can('policy:write')
  const { data, loading, error, reload } = useApi<Policy>('/api/system/policy')
  const [draft, setDraft] = useState<Policy | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (data) setDraft(JSON.parse(JSON.stringify(data)))
  }, [data])

  const dirty = useMemo(
    () => Boolean(data && draft && JSON.stringify(data) !== JSON.stringify(draft)),
    [data, draft],
  )

  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (loading && !draft) return <Loading label="Loading policy" />
  if (!draft) return null

  function setValue(section: string, key: string, value: unknown) {
    setDraft((prev) => (prev ? { ...prev, [section]: { ...prev[section], [key]: value } } : prev))
  }

  async function save() {
    if (!draft) return
    setBusy(true)
    try {
      const payload: Policy = {}
      sections.forEach((section) => {
        if (draft[section]) payload[section] = draft[section]
      })
      await api.put('/api/system/policy', payload)
      toast('Policy saved — agents reloaded')
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {!editable ? (
        <InfoNote icon="lock">Your role can view the policy but not change it.</InfoNote>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg items-start">
        {sections.map((section) => {
          const values = draft[section]
          if (!values) return null
          return (
            <Card key={section} title={titleCase(section)}>
              {SECTION_HELP[section] ? (
                <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
                  {SECTION_HELP[section]}
                </p>
              ) : null}
              <div className="flex flex-col gap-space-sm">
                {Object.entries(values).map(([key, value]) => (
                  <PolicyRow
                    key={key}
                    section={section}
                    name={key}
                    value={value}
                    editable={editable}
                    onChange={(next) => setValue(section, key, next)}
                  />
                ))}
              </div>
            </Card>
          )
        })}
      </div>

      {editable ? (
        <div className="sticky bottom-4 flex justify-end">
          <div className="card px-space-md py-space-sm flex items-center gap-space-md shadow-lg">
            <span className="font-body-sm text-body-sm text-on-surface-variant">
              {dirty ? 'Unsaved changes' : 'All changes saved'}
            </span>
            <button className="btn-secondary" onClick={reload} disabled={!dirty || busy}>Discard</button>
            <button className="btn-primary" onClick={save} disabled={!dirty || busy}>
              {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : null}
              Save policy
            </button>
          </div>
        </div>
      ) : null}
    </>
  )
}

function PolicyRow({
  section,
  name,
  value,
  editable,
  onChange,
}: {
  section: string
  name: string
  value: unknown
  editable: boolean
  onChange: (next: unknown) => void
}) {
  const label = section === 'threat_weights' ? behaviorLabel(name) : titleCase(name)

  if (typeof value === 'boolean') {
    return (
      <div className="flex items-center justify-between gap-space-md">
        <span className="font-body-sm text-body-sm">{label}</span>
        <Toggle checked={value} onChange={onChange} disabled={!editable} />
      </div>
    )
  }

  if (typeof value === 'number') {
    return (
      <div className="flex items-center justify-between gap-space-md">
        <span className="font-body-sm text-body-sm min-w-0 truncate" title={name}>{label}</span>
        <input
          type="number"
          step={Number.isInteger(value) ? 1 : 0.01}
          className="input w-28 text-right"
          value={value}
          disabled={!editable}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
    )
  }

  if (Array.isArray(value)) {
    return (
      <div className="flex flex-col gap-1">
        <span className="label mb-0">{label}</span>
        <input
          className="input"
          value={value.join(', ')}
          disabled={!editable}
          onChange={(e) => onChange(e.target.value.split(',').map((v) => v.trim()).filter(Boolean))}
        />
      </div>
    )
  }

  if (value && typeof value === 'object') {
    return (
      <div className="flex flex-col gap-1 pl-space-sm border-l-2 border-outline-variant/40">
        <span className="label mb-0">{label}</span>
        {Object.entries(value as Record<string, unknown>).map(([key, nested]) => (
          <div key={key} className="flex items-center justify-between gap-space-md">
            <span className="font-body-sm text-body-sm text-on-surface-variant">{titleCase(key)}</span>
            <input
              type={typeof nested === 'number' ? 'number' : 'text'}
              step={typeof nested === 'number' && Number.isInteger(nested) ? 1 : 0.01}
              className="input w-28 text-right"
              value={String(nested)}
              disabled={!editable}
              onChange={(e) =>
                onChange({
                  ...(value as Record<string, unknown>),
                  [key]: typeof nested === 'number' ? Number(e.target.value) : e.target.value,
                })
              }
            />
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="flex items-center justify-between gap-space-md">
      <span className="font-body-sm text-body-sm">{label}</span>
      <input className="input w-48" value={String(value ?? '')} disabled={!editable} onChange={(e) => onChange(e.target.value)} />
    </div>
  )
}

/* ---------------------------------------------------------------- compute */
function ComputePanel() {
  const { status, setServerGpu, busy, clientGpu, setClientGpu } = useGpu()
  const { data: models, loading } = useApi<{
    registered: { id: string; name: string; task: string; framework: string; version: string; status: string; notes: string | null }[]
    runtime: Record<string, any>[]
  }>('/api/system/models', undefined, { pollMs: 20000 })
  const { data: llm, reload: reloadLlm } = useApi<{ ok: boolean; provider: string; model: string | null; enabled: boolean; reason?: string }>(
    '/api/system/llm/health',
  )
  const { can } = useAuth()

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg items-start">
      <Card title="Server inference">
        <div className="flex flex-col gap-space-md">
          <Toggle
            checked={Boolean(status?.gpu_enabled)}
            onChange={(next) => setServerGpu(next).catch(() => toast('Could not change device', 'error'))}
            disabled={!can('system:gpu')}
            busy={busy}
            label="Use the GPU for server-side inference"
            description="Moves every loaded model between CUDA and CPU immediately. Turning it off pushes the NVIDIA models to CPU only."
          />
          <dl className="grid grid-cols-2 gap-space-md">
            <Stat label="Active device" value={status?.active_device ?? '—'} />
            <Stat label="CUDA available" value={status?.cuda_available ? 'yes' : 'no'} />
            <Stat label="Torch" value={status?.torch_version ?? 'not installed'} />
            <Stat label="CUDA build" value={status?.cuda_version ?? '—'} />
          </dl>
          {status?.gpus.map((gpu) => (
            <p key={gpu.index} className="mono text-on-surface-variant">
              GPU {gpu.index}: {gpu.name} · {(gpu.total_memory_mb / 1024).toFixed(1)} GB
              {gpu.driver ? ` · driver ${gpu.driver}` : ''}
            </p>
          ))}
          <Toggle
            checked={clientGpu}
            onChange={setClientGpu}
            label="Use this browser's GPU for rendering"
            description="WebGPU high-performance adapter for the twin and heatmaps. No permission prompt, and it falls back to software when off."
          />
        </div>
      </Card>

      <Card title="Language model">
        {!llm ? (
          <Loading />
        ) : (
          <div className="flex flex-col gap-space-md">
            <dl className="grid grid-cols-2 gap-space-md">
              <Stat label="Provider" value={llm.provider} />
              <Stat label="Model" value={llm.model ?? '—'} />
              <Stat label="Reachable" value={llm.ok ? 'yes' : 'no'} />
              <Stat label="Configured" value={llm.enabled ? 'yes' : 'no'} />
            </dl>
            {!llm.enabled ? (
              <InfoNote icon="offline_bolt">
                No provider configured. Incident summaries, reports and assistant answers are composed
                deterministically from the evidence — fully offline and unable to invent anything. Set
                <code className="mono"> SENTINEL_LLM_PROVIDER</code> and related variables in the backend
                <code className="mono"> .env</code> to enable natural phrasing.
              </InfoNote>
            ) : llm.reason ? (
              <ErrorNote message={llm.reason} />
            ) : null}
            <button className="btn-secondary self-start" onClick={reloadLlm}>
              <Icon name="refresh" className="text-[16px]" />
              Re-check
            </button>
          </div>
        )}
      </Card>

      <Card className="lg:col-span-2" title="Model registry" bodyClassName="p-0">
        {loading && !models ? (
          <Loading />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Name</th>
                  <th>Framework</th>
                  <th>Status</th>
                  <th>Running backend</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {(models?.registered ?? []).map((model) => {
                  const runtime = (models?.runtime ?? []).find((r) => r.task === model.task)
                  return (
                    <tr key={model.id}>
                      <td className="mono">{model.task}</td>
                      <td className="truncate max-w-[16rem]">{model.name}</td>
                      <td className="mono text-on-surface-variant">{model.framework}</td>
                      <td className="mono">{model.status}</td>
                      <td className="mono text-on-surface-variant truncate max-w-[16rem]">
                        {runtime?.backend ?? '—'}
                      </td>
                      <td className="text-on-surface-variant max-w-md">{model.notes}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
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

/* -------------------------------------------------------------- workflows */
interface Workflow {
  id: string
  name: string
  description: string | null
  trigger: string
  webhook_url: string
  method: string
  conditions: Record<string, unknown>
  enabled: boolean
  run_count: number
  failure_count: number
  last_run_at: string | null
}

function WorkflowsPanel() {
  const { can } = useAuth()
  const editable = can('workflow:write')
  const { data, loading, error, reload } = useApi<{
    items: Workflow[]
    available_triggers: string[]
    dispatcher: { enabled: boolean; dispatched: number; failed: number; queue_depth: number; running: boolean }
    condition_help: Record<string, string>
  }>('/api/admin/workflows', undefined, { pollMs: 20000 })

  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({
    name: '', trigger: 'incident_created', webhook_url: '', description: '',
    min_severity: '', min_risk_score: '',
  })

  async function create() {
    try {
      const conditions: Record<string, unknown> = {}
      if (form.min_severity) conditions.min_severity = form.min_severity
      if (form.min_risk_score) conditions.min_risk_score = Number(form.min_risk_score)
      await api.post('/api/admin/workflows', {
        name: form.name,
        trigger: form.trigger,
        webhook_url: form.webhook_url,
        description: form.description || null,
        conditions,
      })
      toast('Workflow binding created')
      setOpen(false)
      setForm({ name: '', trigger: 'incident_created', webhook_url: '', description: '', min_severity: '', min_risk_score: '' })
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function test(workflow: Workflow) {
    try {
      const result = await api.post<{ ok: boolean; status_code?: number; error?: string }>(
        `/api/admin/workflows/${workflow.id}/test`,
      )
      toast(result.ok ? `Webhook responded ${result.status_code}` : `Failed: ${result.error ?? result.status_code}`, result.ok ? 'ok' : 'error')
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function remove(workflow: Workflow) {
    if (!confirm(`Delete the binding "${workflow.name}"?`)) return
    await api.delete(`/api/admin/workflows/${workflow.id}`)
    toast('Binding deleted')
    reload()
  }

  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (loading && !data) return <Loading />

  return (
    <>
      <InfoNote icon="conversion_path">
        n8n sits around Eagles Eye, not inside the vision loop. Each binding forwards a matching event
        to a webhook on a background thread, so a slow or unreachable n8n never stalls a camera.
        {data ? (
          <span className="block mt-1 mono text-outline">
            dispatcher {data.dispatcher.running ? 'running' : 'stopped'} · {data.dispatcher.dispatched}{' '}
            dispatched · {data.dispatcher.failed} failed · queue {data.dispatcher.queue_depth}
          </span>
        ) : null}
      </InfoNote>

      <Card
        title="Workflow bindings"
        bodyClassName="p-0"
        actions={
          editable ? (
            <button className="btn-primary btn-xs" onClick={() => setOpen(true)}>
              <Icon name="add" className="text-[14px]" />
              New binding
            </button>
          ) : null
        }
      >
        {!data || data.items.length === 0 ? (
          <Empty
            icon="webhook"
            title="No bindings configured"
            hint="Point a trigger at an n8n webhook to create incidents, notify a team or generate a report automatically."
          />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Trigger</th>
                  <th>Conditions</th>
                  <th>Webhook</th>
                  <th>Runs</th>
                  <th>Last run</th>
                  <th>Enabled</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.items.map((workflow) => (
                  <tr key={workflow.id}>
                    <td>{workflow.name}</td>
                    <td className="mono">{workflow.trigger}</td>
                    <td className="mono text-on-surface-variant truncate max-w-[12rem]">
                      {Object.keys(workflow.conditions).length
                        ? Object.entries(workflow.conditions).map(([k, v]) => `${k}=${v}`).join(', ')
                        : 'any'}
                    </td>
                    <td className="mono text-on-surface-variant truncate max-w-[14rem]" title={workflow.webhook_url}>
                      {workflow.webhook_url}
                    </td>
                    <td className="mono tabular-nums">
                      {workflow.run_count}
                      {workflow.failure_count ? <span className="text-red-700"> / {workflow.failure_count} failed</span> : null}
                    </td>
                    <td className="mono text-on-surface-variant">
                      {workflow.last_run_at ? relativeTime(workflow.last_run_at) : 'never'}
                    </td>
                    <td className="mono">{workflow.enabled ? 'yes' : 'no'}</td>
                    <td className="w-32">
                      {editable ? (
                        <div className="flex gap-1">
                          <button className="btn-secondary btn-xs" onClick={() => test(workflow)}>Test</button>
                          <button className="btn-ghost btn-xs" onClick={() => remove(workflow)}>
                            <Icon name="delete" className="text-[14px]" />
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

      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title="New workflow binding"
        footer={
          <>
            <button className="btn-secondary" onClick={() => setOpen(false)}>Cancel</button>
            <button className="btn-primary" onClick={create} disabled={!form.name || !form.webhook_url}>
              Create
            </button>
          </>
        }
      >
        <div className="flex flex-col gap-space-md">
          <Field label="Name" required>
            <input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </Field>
          <Field label="Trigger" required>
            <select className="input" value={form.trigger} onChange={(e) => setForm({ ...form, trigger: e.target.value })}>
              {(data?.available_triggers ?? []).map((trigger) => (
                <option key={trigger} value={trigger}>{titleCase(trigger)}</option>
              ))}
            </select>
          </Field>
          <Field label="Webhook URL" required hint="Your n8n webhook, e.g. http://localhost:5678/webhook/sentinel">
            <input className="input" value={form.webhook_url} onChange={(e) => setForm({ ...form, webhook_url: e.target.value })} />
          </Field>
          <div className="grid grid-cols-2 gap-space-md">
            <Field label="Minimum severity">
              <select className="input" value={form.min_severity} onChange={(e) => setForm({ ...form, min_severity: e.target.value })}>
                <option value="">Any</option>
                {['low', 'medium', 'high', 'critical'].map((s) => (
                  <option key={s} value={s}>{titleCase(s)}</option>
                ))}
              </select>
            </Field>
            <Field label="Minimum risk score">
              <input
                type="number" className="input" value={form.min_risk_score}
                onChange={(e) => setForm({ ...form, min_risk_score: e.target.value })}
                placeholder="0-100"
              />
            </Field>
          </div>
        </div>
      </Modal>
    </>
  )
}

/* ------------------------------------------------------------------ users */
function UsersPanel() {
  const { can, user } = useAuth()
  const isAdmin = can('*')
  const { data, loading, error, reload } = useApi<{ items: any[] }>('/api/auth/users', undefined, { enabled: can('audit:read') })
  const { data: roles } = useApi<{ roles: { role: string; permissions: string[] }[] }>('/api/auth/roles')
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({ username: '', password: '', full_name: '', role: 'operator', post: '' })

  if (!can('audit:read')) {
    return <InfoNote icon="lock">Your role cannot view the operator list.</InfoNote>
  }
  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (loading && !data) return <Loading />

  async function create() {
    try {
      await api.post('/api/auth/users', form)
      toast(`${form.username} created`)
      setOpen(false)
      setForm({ username: '', password: '', full_name: '', role: 'operator', post: '' })
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  return (
    <>
      <Card
        title="Operators"
        bodyClassName="p-0"
        actions={
          isAdmin ? (
            <button className="btn-primary btn-xs" onClick={() => setOpen(true)}>
              <Icon name="person_add" className="text-[14px]" />
              Add operator
            </button>
          ) : null
        }
      >
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Username</th>
                <th>Name</th>
                <th>Role</th>
                <th>Post</th>
                <th>Active</th>
                <th>Last sign-in</th>
              </tr>
            </thead>
            <tbody>
              {(data?.items ?? []).map((row) => (
                <tr key={row.id}>
                  <td className="mono">{row.username}{row.username === user?.username ? ' (you)' : ''}</td>
                  <td>{row.full_name}</td>
                  <td className="mono">{row.role}</td>
                  <td className="text-on-surface-variant">{row.post ?? '—'}</td>
                  <td className="mono">{row.active ? 'yes' : 'no'}</td>
                  <td className="mono text-on-surface-variant">
                    {row.last_login_at ? relativeTime(row.last_login_at) : 'never'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Roles and permissions" bodyClassName="p-0">
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th className="w-32">Role</th>
                <th>Permissions</th>
              </tr>
            </thead>
            <tbody>
              {(roles?.roles ?? []).map((role) => (
                <tr key={role.role}>
                  <td className="mono font-semibold">{role.role}</td>
                  <td className="mono text-on-surface-variant">
                    {role.permissions.includes('*') ? 'all permissions' : role.permissions.join(' · ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title="Add operator"
        footer={
          <>
            <button className="btn-secondary" onClick={() => setOpen(false)}>Cancel</button>
            <button className="btn-primary" onClick={create} disabled={!form.username || form.password.length < 8}>
              Create
            </button>
          </>
        }
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
          <Field label="Username" required>
            <input className="input" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
          </Field>
          <Field label="Password" required hint="At least 8 characters.">
            <input type="password" className="input" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          </Field>
          <Field label="Full name">
            <input className="input" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
          </Field>
          <Field label="Role">
            <select className="input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
              {(roles?.roles ?? []).map((role) => (
                <option key={role.role} value={role.role}>{role.role}</option>
              ))}
            </select>
          </Field>
          <Field label="Post">
            <input className="input" value={form.post} onChange={(e) => setForm({ ...form, post: e.target.value })} />
          </Field>
        </div>
      </Modal>
    </>
  )
}
