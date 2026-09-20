/**
 * Application shell: fixed header, grouped sidebar, content area.
 *
 * Layout and navigation grouping follow the Stitch export; the counters,
 * status pills and clock are all live values from the API rather than the
 * fixed numbers in the static mockups.
 */
import { ReactNode, useEffect, useMemo, useState } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useApi, useConnection, useLiveEvents } from '../lib/hooks'
import { GpuToggle } from './GpuToggle'
import { Icon } from './ui'

interface NavItem {
  to: string
  label: string
  icon: string
  permission?: string
  badge?: 'incidents' | 'alerts' | 'approvals' | 'health'
}

interface NavGroup {
  label: string
  items: NavItem[]
}

export const NAV: NavGroup[] = [
  {
    label: 'Operations',
    items: [
      { to: '/', label: 'Command Center', icon: 'desktop_windows' },
      { to: '/cameras', label: 'Live Cameras', icon: 'videocam' },
      { to: '/incidents', label: 'Active Incidents', icon: 'warning', badge: 'incidents' },
      { to: '/alerts', label: 'Alerts', icon: 'emergency_home', badge: 'alerts' },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { to: '/people', label: 'People & Tracking', icon: 'person_search', permission: 'track:read' },
      { to: '/watchlist', label: 'Person of Interest', icon: 'how_to_reg', permission: 'watchlist:read' },
      { to: '/behavior', label: 'Behavior Intelligence', icon: 'psychology' },
      { to: '/crowd', label: 'Crowd Intelligence', icon: 'groups' },
      { to: '/objects', label: 'Object Intelligence', icon: 'category' },
      { to: '/threat', label: 'Threat Assessment', icon: 'shield' },
    ],
  },
  {
    label: 'Investigation',
    items: [
      { to: '/forensic', label: 'Forensic Search', icon: 'manage_search' },
      { to: '/assistant', label: 'AI Assistant', icon: 'auto_awesome' },
      { to: '/cases', label: 'Cases', icon: 'folder_open', permission: 'case:read' },
      { to: '/evidence', label: 'Evidence Locker', icon: 'lock_open', permission: 'evidence:read' },
      { to: '/reports', label: 'Reports', icon: 'description', permission: 'report:read' },
    ],
  },
  {
    label: 'Spatial & AR',
    items: [
      { to: '/twin', label: 'Digital Twin', icon: '3d_rotation' },
      { to: '/ar', label: 'AR Field Vision', icon: 'view_in_ar' },
    ],
  },
  {
    label: 'Automation & Admin',
    items: [
      { to: '/agents', label: 'Agent Orchestrator', icon: 'account_tree' },
      { to: '/workflows', label: 'Workflows (n8n)', icon: 'conversion_path', permission: 'workflow:read' },
      { to: '/approvals', label: 'Approvals', icon: 'how_to_vote', badge: 'approvals' },
      { to: '/health', label: 'System Health', icon: 'monitor_heart', badge: 'health' },
      { to: '/privacy', label: 'Privacy & Audit', icon: 'verified_user' },
      { to: '/settings', label: 'Settings', icon: 'settings' },
    ],
  },
]

interface Counters {
  incidents: number
  alerts: number
  approvals: number
  health: string
}

export function AppShell({ children }: { children: ReactNode }) {
  const { user, logout, can } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const connected = useConnection()
  const [clock, setClock] = useState(() => new Date())
  const [query, setQuery] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const { data: summary, refresh: refreshSummary } = useApi<{ open: number }>(
    '/api/incidents/summary',
    { since_hours: 24 },
    { pollMs: 30000 },
  )
  const { data: alerts, refresh: refreshAlerts } = useApi<{ total: number }>(
    '/api/alerts',
    { unacknowledged_only: true, limit: 1 },
    { pollMs: 30000 },
  )
  const { data: approvals, refresh: refreshApprovals } = useApi<{ total: number }>(
    '/api/admin/approvals',
    { status_filter: 'pending', limit: 1 },
    { pollMs: 45000 },
  )
  const { data: info } = useApi<{
    counts: { cameras_online: number; cameras_total: number; cameras_running: number }
    compute: { active_device: string }
  }>('/api/system/info', undefined, { pollMs: 20000 })

  useLiveEvents(['incident', 'approval'], () => {
    refreshSummary()
    refreshAlerts()
    refreshApprovals()
  })

  useEffect(() => {
    const timer = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    setSidebarOpen(false)
  }, [location.pathname])

  const counters: Counters = {
    incidents: summary?.open ?? 0,
    alerts: alerts?.total ?? 0,
    approvals: approvals?.total ?? 0,
    health: info ? `${info.counts.cameras_online}/${info.counts.cameras_total}` : '—',
  }

  const groups = useMemo(
    () =>
      NAV.map((group) => ({
        ...group,
        items: group.items.filter((item) => !item.permission || can(item.permission)),
      })).filter((group) => group.items.length > 0),
    [can],
  )

  function submitSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    navigate(`/forensic?q=${encodeURIComponent(query.trim())}`)
    setQuery('')
  }

  return (
    <div className="min-h-screen bg-background">
      {/* ---------------------------------------------------------- header */}
      <header className="fixed top-0 inset-x-0 h-14 bg-surface-container-lowest border-b border-outline-variant/40 z-50 px-space-lg flex items-center justify-between gap-space-lg shadow-sm">
        <div className="flex items-center gap-space-md shrink-0">
          <button
            className="btn-ghost btn-xs xl:hidden"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="Toggle navigation"
          >
            <Icon name="menu" className="text-[20px]" />
          </button>
          <div className="w-8 h-8 rounded-lg bg-primary-container flex items-center justify-center shrink-0">
            <Icon name="shield_person" className="text-on-primary text-[20px]" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="font-headline-sm text-headline-sm font-bold tracking-wider text-on-surface">
              EAGLES EYE
            </span>
            <span className="font-label-sm text-label-sm uppercase tracking-widest text-on-surface-variant hidden sm:block">
              See more. Understand more. Act faster.
            </span>
          </div>
        </div>

        <form onSubmit={submitSearch} className="flex-1 max-w-xl hidden md:block">
          <div className="relative flex items-center">
            <Icon name="smart_toy" className="absolute left-2.5 text-[18px] text-on-surface-variant" />
            <input
              className="input pl-9"
              placeholder="Ask Eagles Eye — e.g. unattended objects in the last hour"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        </form>

        <div className="flex items-center gap-space-sm shrink-0">
          <span
            className={`pill hidden lg:inline-flex ${
              connected
                ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                : 'bg-amber-50 border-amber-300 text-amber-900'
            }`}
            title={connected ? 'Live event stream connected' : 'Reconnecting to the live event stream'}
          >
            <span className={`pip ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
            {connected ? 'Live' : 'Reconnecting'}
          </span>

          <span className="pill hidden lg:inline-flex bg-surface-container border-outline-variant/60 text-on-surface">
            <Icon name="videocam" className="text-[14px] text-primary-container" />
            {counters.health} cameras
          </span>

          <GpuToggle compact />

          <span className="mono text-on-surface-variant hidden sm:block tabular-nums">
            {clock.toISOString().slice(11, 19)} UTC
          </span>

          <NavLink to="/alerts" className="relative btn-ghost btn-xs" aria-label="Alerts">
            <Icon name="notifications" className="text-[20px]" />
            {counters.alerts > 0 ? (
              <span className="absolute -top-1 -right-1 min-w-4 h-4 px-1 rounded-full bg-error text-on-error mono font-bold flex items-center justify-center">
                {counters.alerts > 99 ? '99+' : counters.alerts}
              </span>
            ) : null}
          </NavLink>

          <div className="flex items-center gap-2 pl-1">
            <div className="w-7 h-7 rounded-full bg-primary-container flex items-center justify-center shrink-0">
              <Icon name="person" className="text-on-primary text-[16px]" />
            </div>
            <div className="hidden lg:flex flex-col leading-tight min-w-0">
              <span className="font-headline-sm text-headline-sm text-on-surface truncate max-w-[10rem]">
                {user?.full_name || user?.username}
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant capitalize">{user?.role}</span>
            </div>
            <button
              className="btn-ghost btn-xs"
              onClick={() => logout().then(() => navigate('/login'))}
              title="Sign out"
            >
              <Icon name="logout" className="text-[16px]" />
            </button>
          </div>
        </div>
      </header>

      {/* --------------------------------------------------------- sidebar */}
      <aside
        className={`fixed left-0 top-14 bottom-0 w-64 bg-surface-container-lowest border-r border-outline-variant/40 z-40
                    overflow-y-auto transition-transform xl:translate-x-0 ${
                      sidebarOpen ? 'translate-x-0' : '-translate-x-full'
                    }`}
      >
        <nav className="p-space-sm flex flex-col gap-space-md">
          {groups.map((group) => (
            <div key={group.label} className="flex flex-col gap-0.5">
              <div className="px-3 py-1 font-label-sm text-label-sm uppercase tracking-wider text-outline font-semibold">
                {group.label}
              </div>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg transition-colors font-headline-sm text-headline-sm ${
                      isActive
                        ? 'bg-primary-container text-on-primary font-semibold shadow-sm'
                        : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
                    }`
                  }
                >
                  <span className="flex items-center gap-2.5 min-w-0">
                    <Icon name={item.icon} className="text-[19px] shrink-0" />
                    <span className="truncate">{item.label}</span>
                  </span>
                  <NavBadge kind={item.badge} counters={counters} />
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      {sidebarOpen ? (
        <div
          className="fixed inset-0 top-14 bg-slate-900/30 z-30 xl:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      ) : null}

      {/* --------------------------------------------------------- content */}
      <main className="pt-14 xl:pl-64 min-h-screen">
        <div className="p-gutter-lg flex flex-col gap-space-lg max-w-[1900px]">{children}</div>
      </main>
    </div>
  )
}

function NavBadge({ kind, counters }: { kind?: NavItem['badge']; counters: Counters }) {
  if (!kind) return null
  if (kind === 'health') {
    return <span className="mono text-on-surface-variant">{counters.health}</span>
  }
  const value = counters[kind]
  if (!value) return null
  const tone =
    kind === 'incidents'
      ? 'bg-error-container text-on-error-container'
      : kind === 'alerts'
        ? 'bg-amber-100 text-amber-900'
        : 'bg-surface-container-high text-on-surface'
  return <span className={`mono px-1.5 py-0.5 rounded-full font-bold ${tone}`}>{value}</span>
}
