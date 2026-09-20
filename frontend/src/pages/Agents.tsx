/** Agent Orchestrator — the full specification roster, with live health. */
import { useEffect } from 'react'
import { Card, Empty, ErrorNote, Icon, Kpi, Loading, PageHeader, Toggle, toast } from '../components/ui'
import { ApiError, api, type AgentRow } from '../lib/api'
import { useAuth } from '../lib/auth'
import { number, relativeTime } from '../lib/format'
import { useApi } from '../lib/hooks'

interface Roster {
  agents: AgentRow[]
  frames_processed: number
  findings_emitted: number
  incidents_opened: number
  open_incidents: number
  device: string
  policy_loaded_at: string
}

const LOCATION_LABEL: Record<string, string> = {
  pipeline: 'Capture pipeline',
  orchestrator: 'Per-frame agent',
  reasoner: 'Reasoning layer',
}

const AGENT_ICON: Record<string, string> = {
  vision: 'visibility',
  tracking: 'route',
  reid: 'link',
  appearance: 'change_circle',
  occlusion: 'masks',
  behavior: 'psychology',
  relationship: 'follow_the_signs',
  crowd: 'groups',
  object: 'luggage',
  threat: 'shield',
  commander: 'military_tech',
  forensic: 'manage_search',
  watchlist: 'person_search',
  spatial: '3d_rotation',
  privacy: 'verified_user',
}

export default function AgentsPage() {
  const { can } = useAuth()
  const { data, loading, error, refresh, reload } = useApi<Roster>(
    '/api/system/agents', undefined, { pollMs: 5000 },
  )

  useEffect(() => {
    document.title = 'Agent Orchestrator · Eagles Eye'
  }, [])

  async function toggle(agent: AgentRow, enabled: boolean) {
    try {
      await api.post(`/api/system/agents/${agent.name}/toggle`, { enabled })
      toast(`${agent.name} ${enabled ? 'enabled' : 'disabled'}`)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (loading && !data) return <Loading label="Loading agent roster" />
  if (!data) return <Empty icon="account_tree" title="No agent data" />

  const failing = data.agents.filter((a) => a.last_error)
  const grouped = ['pipeline', 'orchestrator', 'reasoner'].map((location) => ({
    location,
    agents: data.agents.filter((a) => a.location === location),
  }))

  return (
    <>
      <PageHeader
        title="Agent Orchestrator"
        subtitle="Twelve specialised agents plus the privacy and spatial layers, coordinated per frame"
        actions={
          <button className="btn-ghost btn-xs" onClick={refresh}>
            <Icon name="refresh" className="text-[16px]" />
          </button>
        }
      />

      <section className="grid grid-cols-2 lg:grid-cols-5 gap-space-md">
        <Kpi label="Frames analysed" value={number(data.frames_processed)} />
        <Kpi label="Findings emitted" value={number(data.findings_emitted)} />
        <Kpi label="Incidents opened" value={number(data.incidents_opened)} />
        <Kpi label="Open now" value={number(data.open_incidents)} />
        <Kpi
          label="Agent errors"
          value={failing.length}
          tone={failing.length ? 'critical' : 'good'}
          footer={data.device}
        />
      </section>

      {failing.length > 0 ? (
        <div className="card p-space-md bg-red-50 border-red-200">
          <p className="font-headline-sm text-headline-sm text-red-900 mb-1">Agents reporting errors</p>
          {failing.map((agent) => (
            <p key={agent.name} className="mono text-red-800">
              {agent.name}: {agent.last_error}
            </p>
          ))}
        </div>
      ) : null}

      {grouped.map((group) =>
        group.agents.length === 0 ? null : (
          <Card key={group.location} title={LOCATION_LABEL[group.location] ?? group.location} bodyClassName="p-0">
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th className="w-14">Spec</th>
                    <th>Agent</th>
                    <th>Responsibility</th>
                    <th>Backend</th>
                    <th>Last run</th>
                    <th>Runs</th>
                    <th>Findings</th>
                    <th className="w-24">Enabled</th>
                  </tr>
                </thead>
                <tbody>
                  {group.agents.map((agent) => (
                    <tr key={`${agent.location}-${agent.name}`}>
                      <td className="mono text-on-surface-variant">
                        {agent.spec_id ? `#${agent.spec_id}` : '—'}
                      </td>
                      <td>
                        <span className="flex items-center gap-2">
                          <Icon
                            name={AGENT_ICON[agent.name] ?? 'smart_toy'}
                            className="text-[16px] text-primary-container"
                          />
                          <span className="font-headline-sm text-headline-sm capitalize">{agent.name}</span>
                        </span>
                      </td>
                      <td className="text-on-surface-variant max-w-md">{agent.description}</td>
                      <td className="mono text-on-surface-variant truncate max-w-[12rem]">
                        {agent.backend ?? '—'}
                      </td>
                      <td className="mono tabular-nums">
                        {agent.last_run_ms !== undefined ? `${agent.last_run_ms.toFixed(1)} ms` : '—'}
                      </td>
                      <td className="mono tabular-nums">{number(agent.total_runs ?? 0)}</td>
                      <td className="mono tabular-nums">{number(agent.total_findings ?? 0)}</td>
                      <td>
                        {group.location === 'orchestrator' && can('policy:write') ? (
                          <Toggle checked={agent.enabled} onChange={(next) => toggle(agent, next)} />
                        ) : (
                          <span className="mono text-on-surface-variant">
                            {agent.enabled ? 'yes' : 'no'}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ),
      )}

      <p className="font-body-sm text-body-sm text-on-surface-variant">
        Policy last loaded {relativeTime(data.policy_loaded_at)}. Thresholds and weights are edited in
        Settings and take effect without a restart.
      </p>
    </>
  )
}
