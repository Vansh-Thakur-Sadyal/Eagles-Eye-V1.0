import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { Icon, Toaster } from './components/ui'
import { useAuth } from './lib/auth'

import AgentsPage from './pages/Agents'
import AlertsPage from './pages/Alerts'
import ArPage from './pages/ArField'
import AssistantPage from './pages/Assistant'
import BehaviorPage from './pages/Behavior'
import CamerasPage from './pages/Cameras'
import CasesPage from './pages/Cases'
import CommandCenterPage from './pages/CommandCenter'
import CrowdPage from './pages/Crowd'
import EvidencePage from './pages/Evidence'
import ForensicPage from './pages/Forensic'
import HealthPage from './pages/Health'
import IncidentDetailPage from './pages/IncidentDetail'
import IncidentsPage from './pages/Incidents'
import ObjectsPage from './pages/Objects'
import PeoplePage from './pages/People'
import PrivacyPage from './pages/Privacy'
import ApprovalsPage from './pages/Approvals'
import ReportsPage from './pages/Reports'
import SettingsPage from './pages/Settings'
import ThreatPage from './pages/Threat'
import TwinPage from './pages/Twin'
import WatchlistPage from './pages/Watchlist'
import WatchlistDetailPage from './pages/WatchlistDetail'

function Protected({ children }: { children: JSX.Element }) {
  const { user, loading } = useAuth()
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center gap-2 text-on-surface-variant">
        <Icon name="progress_activity" className="animate-spin text-[20px]" />
        <span className="font-body-md text-body-md">Restoring session…</span>
      </div>
    )
  }
  if (!user) return <SignInGate />
  return children
}

/** Nobody sees the dashboard without a verified session.
 *
 * The shell behind is blurred rather than hidden, so it is obvious there is a
 * product here, while no data is readable - the API would refuse it anyway,
 * this is the visual half of the same rule. OK sends them to the portal.
 */
function SignInGate() {
  const portal = '/portal/sign-in.html'
  return (
    <div className="min-h-screen relative overflow-hidden">
      <div aria-hidden className="absolute inset-0 blur-md opacity-40 pointer-events-none select-none">
        <div className="p-8 grid grid-cols-3 gap-4">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="h-40 rounded-xl bg-surface-container-high border border-outline-variant" />
          ))}
        </div>
      </div>
      <div className="absolute inset-0 bg-surface/70 backdrop-blur-sm" />
      <div className="relative min-h-screen flex items-center justify-center p-6">
        <div className="max-w-md w-full rounded-2xl border border-outline-variant bg-surface p-8 text-center shadow-lg">
          <Icon name="lock" className="text-[32px] text-primary" />
          <h1 className="font-headline-lg text-headline-lg mt-2">Please sign in or sign up</h1>
          <p className="font-body-md text-body-md text-on-surface-variant mt-2">
            The Eagles Eye dashboard is available to verified accounts only. You will be
            taken to the sign-in page.
          </p>
          <button className="btn-primary w-full mt-6" onClick={() => { window.location.href = portal }}>
            OK
          </button>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-4">
            No account yet? <a className="underline" href="/portal/sign-up.html">Register your site</a>
          </p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/login" element={<SignInGate />} />
        <Route
          path="/*"
          element={
            <Protected>
              <AppShell>
                <Routes>
                  <Route path="/" element={<CommandCenterPage />} />
                  <Route path="/cameras" element={<CamerasPage />} />
                  <Route path="/incidents" element={<IncidentsPage />} />
                  <Route path="/incidents/:incidentId" element={<IncidentDetailPage />} />
                  <Route path="/alerts" element={<AlertsPage />} />
                  <Route path="/people" element={<PeoplePage />} />
                  <Route path="/watchlist" element={<WatchlistPage />} />
                  <Route path="/watchlist/:subjectId" element={<WatchlistDetailPage />} />
                  <Route path="/behavior" element={<BehaviorPage />} />
                  <Route path="/crowd" element={<CrowdPage />} />
                  <Route path="/objects" element={<ObjectsPage />} />
                  <Route path="/threat" element={<ThreatPage />} />
                  <Route path="/forensic" element={<ForensicPage />} />
                  <Route path="/assistant" element={<AssistantPage />} />
                  <Route path="/cases" element={<CasesPage />} />
                  <Route path="/evidence" element={<EvidencePage />} />
                  <Route path="/reports" element={<ReportsPage />} />
                  <Route path="/twin" element={<TwinPage />} />
                  <Route path="/ar" element={<ArPage />} />
                  <Route path="/agents" element={<AgentsPage />} />
                  <Route path="/workflows" element={<SettingsPage initialTab="workflows" />} />
                  <Route path="/approvals" element={<ApprovalsPage />} />
                  <Route path="/health" element={<HealthPage />} />
                  <Route path="/privacy" element={<PrivacyPage />} />
                  <Route path="/settings" element={<SettingsPage />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </AppShell>
            </Protected>
          }
        />
      </Routes>
      <Toaster />
    </>
  )
}

function NotFound() {
  return (
    <div className="card p-space-xl flex flex-col items-center gap-2 text-center">
      <Icon name="explore_off" className="text-[32px] text-outline-variant" />
      <h1 className="font-headline-lg text-headline-lg">Screen not found</h1>
      <p className="font-body-sm text-body-sm text-on-surface-variant">
        Pick a destination from the navigation on the left.
      </p>
    </div>
  )
}
