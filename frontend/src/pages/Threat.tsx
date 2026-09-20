/** Threat Assessment — ranked risk plus a what-if simulator over the live weights. */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Loading, Meter, PageHeader, SeverityPill, toast } from '../components/ui'
import { ApiError, api } from '../lib/api'
import { behaviorIcon, behaviorLabel, confidence, riskBand, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface Weights {
  weights: Record<string, number>
  bands: Record<string, number>
  zero_weight_behaviors: string[]
  note: string
}

interface Ranked {
  items: {
    id: string
    title: string
    risk_score: number
    severity: string
    camera_id: string | null
    zone_id: string | null
    dominant_factor: string | null
    explanation: string | null
    started_at: string
  }[]
}

interface Simulation {
  score: number
  band: string
  explanation: string
  confidence: number
  dominant_factor: string | null
  factors: {
    behavior: string
    weight: number
    confidence: number
    contribution: number
    counted: boolean
    note: string | null
  }[]
}

export default function ThreatPage() {
  const { data: weights } = useApi<Weights>('/api/intel/threat/weights')
  const { data: ranked, loading, error, refresh } = useApi<Ranked>(
    '/api/intel/threat/ranked', { limit: 20 }, { pollMs: 15000 },
  )

  const [selected, setSelected] = useState<Record<string, number>>({})
  const [zoneWeight, setZoneWeight] = useState(1.0)
  const [simulation, setSimulation] = useState<Simulation | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    document.title = 'Threat Assessment · Eagles Eye'
  }, [])

  const weighted = useMemo(
    () => Object.entries(weights?.weights ?? {}).filter(([, value]) => value > 0).sort((a, b) => b[1] - a[1]),
    [weights],
  )
  const zeroWeighted = useMemo(
    () => Object.entries(weights?.weights ?? {}).filter(([, value]) => value === 0),
    [weights],
  )

  async function simulate() {
    if (Object.keys(selected).length === 0) {
      toast('Select at least one signal', 'error')
      return
    }
    setBusy(true)
    try {
      setSimulation(
        await api.post<Simulation>('/api/intel/threat/simulate', {
          behaviors: selected,
          zone_risk_weight: zoneWeight,
        }),
      )
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageHeader
        title="Threat Assessment"
        subtitle="Explainable, weighted, saturating risk — never a bare number"
      />

      <InfoNote icon="balance">
        {weights?.note ??
          'Observability signals carry weight 0 by design: a covered face is ordinary behaviour and must never raise a risk score.'}
      </InfoNote>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" title="Highest risk right now" bodyClassName="p-0">
          {error ? (
            <div className="p-space-md"><ErrorNote message={error} onRetry={refresh} /></div>
          ) : loading && !ranked ? (
            <Loading />
          ) : !ranked || ranked.items.length === 0 ? (
            <Empty icon="shield" title="No open incidents" hint="Ranked risk appears as soon as an incident is raised." />
          ) : (
            <ul className="divide-y divide-outline-variant/25">
              {ranked.items.map((item) => (
                <li key={item.id} className="p-space-md">
                  <div className="flex items-start justify-between gap-space-md">
                    <div className="min-w-0 flex-1">
                      <Link to={`/incidents/${item.id}`} className="font-headline-sm text-headline-sm hover:text-primary-container">
                        {item.title}
                      </Link>
                      <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2">
                        {item.explanation}
                      </p>
                      <p className="mono text-outline mt-0.5">
                        {item.camera_id ?? '—'}
                        {item.dominant_factor ? ` · dominant: ${behaviorLabel(item.dominant_factor)}` : ''}
                      </p>
                    </div>
                    <div className="flex flex-col items-end gap-1 shrink-0 w-28">
                      <span className="font-headline-lg text-headline-lg tabular-nums">
                        {item.risk_score.toFixed(0)}
                      </span>
                      <Meter
                        value={item.risk_score}
                        tone={
                          riskBand(item.risk_score) === 'critical'
                            ? 'bg-red-600'
                            : riskBand(item.risk_score) === 'high'
                              ? 'bg-amber-500'
                              : 'bg-primary-container'
                        }
                      />
                      <SeverityPill severity={item.severity} />
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <div className="flex flex-col gap-space-lg">
          <Card title="Scoring weights" actions={<Link to="/settings" className="btn-ghost btn-xs">Edit</Link>}>
            <ul className="flex flex-col gap-1">
              {weighted.map(([behavior, weight]) => (
                <li key={behavior} className="flex items-center gap-space-sm">
                  <Icon name={behaviorIcon(behavior)} className="text-[15px] text-on-surface-variant shrink-0" />
                  <span className="font-body-sm text-body-sm flex-1 truncate">{behaviorLabel(behavior)}</span>
                  <span className="mono tabular-nums w-6 text-right">{weight}</span>
                </li>
              ))}
            </ul>
            {zeroWeighted.length > 0 ? (
              <div className="mt-space-md pt-space-md border-t border-outline-variant/40">
                <p className="label">Weight 0 by design</p>
                {zeroWeighted.map(([behavior]) => (
                  <p key={behavior} className="mono text-on-surface-variant">{behaviorLabel(behavior)}</p>
                ))}
              </div>
            ) : null}
            {weights ? (
              <div className="mt-space-md pt-space-md border-t border-outline-variant/40">
                <p className="label">Bands</p>
                <p className="mono text-on-surface-variant">
                  {Object.entries(weights.bands).map(([band, value]) => `${band} ≥ ${value}`).join(' · ')}
                </p>
              </div>
            ) : null}
          </Card>

          <Card title="What-if simulator">
            <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
              Score a hypothetical combination against the live weights before changing them.
            </p>
            <div className="flex flex-col gap-space-sm max-h-56 overflow-y-auto pr-1">
              {weighted.map(([behavior]) => {
                const active = behavior in selected
                return (
                  <div key={behavior} className="flex items-center gap-space-sm">
                    <input
                      type="checkbox"
                      checked={active}
                      onChange={(e) =>
                        setSelected((prev) => {
                          const next = { ...prev }
                          if (e.target.checked) next[behavior] = 0.8
                          else delete next[behavior]
                          return next
                        })
                      }
                    />
                    <span className="font-body-sm text-body-sm flex-1 truncate">{behaviorLabel(behavior)}</span>
                    {active ? (
                      <>
                        <input
                          type="range"
                          min={0.1}
                          max={1}
                          step={0.05}
                          value={selected[behavior]}
                          onChange={(e) =>
                            setSelected((prev) => ({ ...prev, [behavior]: Number(e.target.value) }))
                          }
                          className="w-20"
                        />
                        <span className="mono tabular-nums w-9 text-right">
                          {Math.round(selected[behavior] * 100)}%
                        </span>
                      </>
                    ) : null}
                  </div>
                )
              })}
            </div>

            <div className="flex items-center gap-space-sm mt-space-md">
              <span className="label mb-0 shrink-0">Zone weight</span>
              <input
                type="range" min={0.5} max={2} step={0.1}
                value={zoneWeight}
                onChange={(e) => setZoneWeight(Number(e.target.value))}
                className="flex-1"
              />
              <span className="mono tabular-nums w-8 text-right">{zoneWeight.toFixed(1)}</span>
            </div>

            <button className="btn-primary w-full mt-space-md" onClick={simulate} disabled={busy}>
              {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : null}
              Simulate
            </button>

            {simulation ? (
              <div className="mt-space-md pt-space-md border-t border-outline-variant/40 flex flex-col gap-space-sm">
                <div className="flex items-baseline gap-2">
                  <span className="font-display-lg text-display-lg font-bold">{simulation.score.toFixed(0)}</span>
                  <SeverityPill severity={simulation.band} />
                </div>
                <p className="font-body-sm text-body-sm text-on-surface-variant">{simulation.explanation}</p>
                <table className="tbl">
                  <tbody>
                    {simulation.factors
                      .filter((f) => f.counted)
                      .map((factor) => (
                        <tr key={factor.behavior}>
                          <td className="!h-7">{behaviorLabel(factor.behavior)}</td>
                          <td className="!h-7 mono tabular-nums text-right w-16">
                            {factor.contribution.toFixed(1)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </Card>
        </div>
      </div>
    </>
  )
}
