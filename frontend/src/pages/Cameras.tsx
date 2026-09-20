/**
 * Live Cameras — attach, configure, run and watch every source.
 *
 * The attach flow is source-aware: choosing "External webcam" probes the host
 * for real device indices, and choosing "ESP32-CAM" probes the board and tells
 * you which of its endpoints actually answered before you save anything.
 */
import { useEffect, useMemo, useState } from 'react'
import {
  Card,
  Empty,
  ErrorNote,
  Field,
  Icon,
  Loading,
  Modal,
  PageHeader,
  StatusPill,
  Toggle,
  toast,
} from '../components/ui'
import { ApiError, api, streamUrl, type Camera, type EdgeStatus, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface SourceType {
  value: string
  label: string
  help: string
}

interface WebcamDevice {
  index: number
  width: number
  height: number
  fps: number
}

interface Esp32Probe {
  base_url: string
  reachable: boolean
  recommended_mode: string | null
  hint?: string
  endpoints: Record<string, { url: string; ok: boolean; status_code?: number; error?: string; content_type?: string }>
}

const BLANK = {
  name: '',
  source_type: 'webcam',
  source_uri: '0',
  location: '',
  site_id: '',
  zone_id: '',
  username: '',
  password: '',
  latitude: '',
  longitude: '',
  floor: 0,
  orientation_deg: 0,
  field_of_view_deg: 82,
  range_m: 25,
  fps: 12,
  width: 1280,
  height: 720,
  rotation: 0,
  privacy_redaction: null as boolean | null,
  processing_node: '',
}

export default function CamerasPage() {
  const { can } = useAuth()
  const editable = can('camera:write')
  const { data, loading, error, refresh, reload } = useApi<Paged<Camera>>(
    '/api/cameras', { limit: 200 }, { pollMs: 8000 },
  )
  const { data: sourceTypes } = useApi<{ types: SourceType[] }>('/api/cameras/source-types')
  const { data: sites } = useApi<{ items: { id: string; name: string }[] }>('/api/spatial/sites')
  const { data: zones } = useApi<{ items: { id: string; name: string; zone_type: string }[] }>('/api/spatial/zones')

  const [attachOpen, setAttachOpen] = useState(false)
  const [selected, setSelected] = useState<Camera | null>(null)

  useEffect(() => {
    document.title = 'Live Cameras · Eagles Eye'
  }, [])

  const running = data?.items.filter((c) => c.running) ?? []

  async function control(camera: Camera, action: 'start' | 'stop') {
    try {
      await api.post(`/api/cameras/${camera.id}/${action}`)
      toast(`${camera.name} ${action === 'start' ? 'started' : 'stopped'}`)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function remove(camera: Camera) {
    if (!confirm(`Delete "${camera.name}"? Its recorded incidents and events are kept.`)) return
    try {
      await api.delete(`/api/cameras/${camera.id}`)
      toast(`${camera.name} removed`)
      setSelected(null)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Live Cameras"
        subtitle={
          data
            ? `${running.length} of ${data.total} running`
            : 'Attach webcams, ESP32-CAM boards, RTSP streams or video files'
        }
        actions={
          <>
            {editable && data && data.total > 0 ? (
              <>
                <button className="btn-secondary" onClick={() => api.post('/api/cameras/start-all').then(reload)}>
                  <Icon name="play_arrow" className="text-[16px]" />
                  Start all
                </button>
                <button className="btn-secondary" onClick={() => api.post('/api/cameras/stop-all').then(reload)}>
                  <Icon name="stop" className="text-[16px]" />
                  Stop all
                </button>
              </>
            ) : null}
            {editable ? (
              <button className="btn-primary" onClick={() => setAttachOpen(true)}>
                <Icon name="add_a_photo" className="text-[16px]" />
                Attach camera
              </button>
            ) : null}
          </>
        }
      />

      {error ? <ErrorNote message={error} onRetry={reload} /> : null}

      {loading && !data ? (
        <Loading label="Loading cameras" />
      ) : !data || data.items.length === 0 ? (
        <Card>
          <Empty
            icon="videocam_off"
            title="No cameras attached"
            hint="Attach your external webcam or an ESP32-CAM board to begin. A built-in synthetic scene is also available if you want to exercise the pipeline without hardware."
            action={
              editable ? (
                <button className="btn-primary" onClick={() => setAttachOpen(true)}>
                  <Icon name="add_a_photo" className="text-[16px]" />
                  Attach camera
                </button>
              ) : null
            }
          />
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-space-lg">
          {data.items.map((camera) => (
            <CameraCard
              key={camera.id}
              camera={camera}
              editable={editable}
              onControl={control}
              onOpen={() => setSelected(camera)}
            />
          ))}
        </div>
      )}

      <AttachDialog
        open={attachOpen}
        onClose={() => setAttachOpen(false)}
        sourceTypes={sourceTypes?.types ?? []}
        sites={sites?.items ?? []}
        zones={zones?.items ?? []}
        onSaved={() => {
          setAttachOpen(false)
          reload()
        }}
      />

      <CameraDetail
        camera={selected}
        onClose={() => setSelected(null)}
        editable={editable}
        onControl={control}
        onDelete={remove}
        onSaved={reload}
        zones={zones?.items ?? []}
      />
    </>
  )
}

/* ------------------------------------------------------------------ card */
function CameraCard({
  camera,
  editable,
  onControl,
  onOpen,
}: {
  camera: Camera
  editable: boolean
  onControl: (camera: Camera, action: 'start' | 'stop') => void
  onOpen: () => void
}) {
  const runtime = camera.runtime
  return (
    <Card
      title={camera.name}
      bodyClassName="p-0"
      actions={<StatusPill status={runtime?.status ?? camera.status} pulse={camera.running} />}
    >
      <button onClick={onOpen} className="block w-full text-left">
        <div className="relative bg-slate-900 aspect-video overflow-hidden">
          {camera.running ? (
            <img src={streamUrl(camera.id)} alt={`${camera.name} live view`} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center gap-1 text-slate-400">
              <Icon name="videocam_off" className="text-[28px]" />
              <span className="mono">{camera.status_detail || 'not running'}</span>
            </div>
          )}
          <span className="absolute bottom-2 left-2 pill bg-black/55 border-white/20 text-white">
            <Icon name="lan" className="text-[12px]" />
            {camera.source_type}
          </span>
        </div>
      </button>

      <div className="p-space-md flex flex-col gap-space-sm">
        <div className="flex items-center justify-between gap-2 min-w-0">
          <span className="font-body-sm text-body-sm text-on-surface-variant truncate">
            {camera.location || 'No location set'}
          </span>
          <span className="mono text-on-surface-variant shrink-0">
            {runtime ? `${runtime.measured_fps.toFixed(1)} fps` : `${camera.fps} fps target`}
          </span>
        </div>

        <dl className="grid grid-cols-3 gap-2">
          <Metric label="Tracks" value={runtime?.track_count ?? 0} />
          <Metric label="Inference" value={runtime ? `${runtime.inference_ms.toFixed(0)} ms` : '—'} />
          <Metric label="Device" value={runtime?.device ?? '—'} />
        </dl>

        {runtime?.detector ? (
          <p className="mono text-outline truncate" title={runtime.detector.backend}>
            detector: {runtime.detector.backend}
          </p>
        ) : null}
        {runtime ? <ProcessedOn runtime={runtime} /> : null}

        {editable ? (
          <div className="flex items-center gap-space-sm pt-1">
            {camera.running ? (
              <button className="btn-secondary flex-1" onClick={() => onControl(camera, 'stop')}>
                <Icon name="stop" className="text-[16px]" />
                Stop
              </button>
            ) : (
              <button className="btn-primary flex-1" onClick={() => onControl(camera, 'start')}>
                <Icon name="play_arrow" className="text-[16px]" />
                Start
              </button>
            )}
            <button className="btn-secondary" onClick={onOpen}>
              <Icon name="tune" className="text-[16px]" />
            </button>
          </div>
        ) : null}
      </div>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-0">
      <dt className="label mb-0">{label}</dt>
      <dd className="mono text-on-surface truncate">{value}</dd>
    </div>
  )
}

/* ---------------------------------------------------------------- attach */
function AttachDialog({
  open,
  onClose,
  sourceTypes,
  sites,
  zones,
  onSaved,
}: {
  open: boolean
  onClose: () => void
  sourceTypes: SourceType[]
  sites: { id: string; name: string }[]
  zones: { id: string; name: string; zone_type: string }[]
  onSaved: () => void
}) {
  const [form, setForm] = useState({ ...BLANK })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [webcams, setWebcams] = useState<WebcamDevice[] | null>(null)
  const [scanning, setScanning] = useState(false)
  const [probe, setProbe] = useState<Esp32Probe | null>(null)

  useEffect(() => {
    if (open) {
      setForm({ ...BLANK })
      setError(null)
      setProbe(null)
      setWebcams(null)
    }
  }, [open])

  const help = useMemo(
    () => sourceTypes.find((t) => t.value === form.source_type)?.help ?? '',
    [sourceTypes, form.source_type],
  )

  function set<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function chooseSource(value: string) {
    setProbe(null)
    setForm((prev) => ({
      ...prev,
      source_type: value,
      source_uri: value === 'webcam' ? '0' : value === 'synthetic' ? '' : '',
    }))
    if (value === 'webcam') scanWebcams()
  }

  async function scanWebcams() {
    setScanning(true)
    try {
      const result = await api.get<{ devices: WebcamDevice[] }>('/api/cameras/discover/webcams')
      setWebcams(result.devices)
      if (result.devices.length && !result.devices.some((d) => String(d.index) === form.source_uri)) {
        set('source_uri', String(result.devices[0].index))
      }
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setScanning(false)
    }
  }

  async function probeBoard() {
    if (!form.source_uri.trim()) {
      toast('Enter the board address first, e.g. 192.168.1.42', 'error')
      return
    }
    setScanning(true)
    try {
      setProbe(await api.post<Esp32Probe>('/api/cameras/discover/esp32', { address: form.source_uri.trim() }))
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    } finally {
      setScanning(false)
    }
  }

  async function save() {
    setBusy(true)
    setError(null)
    try {
      const { processing_node, ...rest } = form
      const payload: Record<string, unknown> = {
        ...rest,
        meta: processing_node ? { processing_node } : {},
        site_id: form.site_id || null,
        zone_id: form.zone_id || null,
        username: form.username || null,
        password: form.password || null,
        latitude: form.latitude === '' ? null : Number(form.latitude),
        longitude: form.longitude === '' ? null : Number(form.longitude),
        floor: Number(form.floor),
        orientation_deg: Number(form.orientation_deg),
        field_of_view_deg: Number(form.field_of_view_deg),
        range_m: Number(form.range_m),
        fps: Number(form.fps),
        width: Number(form.width),
        height: Number(form.height),
        rotation: Number(form.rotation),
      }
      const created = await api.post<Camera>('/api/cameras', payload)
      toast(`${created.name} attached`)
      try {
        await api.post(`/api/cameras/${created.id}/start`)
      } catch {
        toast('Camera saved, but it did not start. Check the source settings.', 'info')
      }
      onSaved()
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
      title="Attach a camera"
      width="max-w-3xl"
      footer={
        <>
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={save} disabled={busy || !form.name.trim()}>
            {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : null}
            Attach and start
          </button>
        </>
      }
    >
      <div className="flex flex-col gap-space-lg">
        {error ? <ErrorNote message={error} /> : null}

        {/* source picker */}
        <div>
          <span className="label">Source</span>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-space-sm">
            {sourceTypes.map((type) => (
              <button
                key={type.value}
                onClick={() => chooseSource(type.value)}
                className={`flex items-center gap-2 p-space-sm rounded-lg border text-left transition-colors ${
                  form.source_type === type.value
                    ? 'border-primary-container bg-surface-container-low ring-1 ring-primary-container'
                    : 'border-outline-variant hover:bg-surface'
                }`}
              >
                <Icon name={sourceIcon(type.value)} className="text-[18px] text-primary-container shrink-0" />
                <span className="font-headline-sm text-headline-sm truncate">{sourceLabel(type.value)}</span>
              </button>
            ))}
          </div>
          {help ? <p className="font-body-sm text-body-sm text-on-surface-variant mt-1.5">{help}</p> : null}
        </div>

        {/* source-specific */}
        {form.source_type === 'webcam' ? (
          <div className="flex flex-col gap-space-sm">
            <div className="flex items-end gap-space-sm">
              <Field label="Device index" required>
                <input className="input" value={form.source_uri} onChange={(e) => set('source_uri', e.target.value)} />
              </Field>
              <button className="btn-secondary" onClick={scanWebcams} disabled={scanning}>
                {scanning ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : <Icon name="search" className="text-[16px]" />}
                Scan
              </button>
            </div>
            {webcams ? (
              webcams.length === 0 ? (
                <p className="font-body-sm text-body-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
                  No webcam responded. Check that the device is connected, not in use by another
                  application, and that this machine is allowed to access the camera.
                </p>
              ) : (
                <div className="flex flex-wrap gap-space-sm">
                  {webcams.map((device) => (
                    <button
                      key={device.index}
                      onClick={() => set('source_uri', String(device.index))}
                      className={`pill ${
                        form.source_uri === String(device.index)
                          ? 'bg-primary-container border-primary-container text-on-primary'
                          : 'bg-surface-container border-outline-variant text-on-surface'
                      }`}
                    >
                      <Icon name="videocam" className="text-[12px]" />
                      #{device.index} · {device.width}×{device.height}
                    </button>
                  ))}
                </div>
              )
            ) : null}
          </div>
        ) : form.source_type === 'esp32cam' ? (
          <div className="flex flex-col gap-space-sm">
            <div className="flex items-end gap-space-sm">
              <Field label="Board address" required hint="Just the host is enough — the stream and capture endpoints are found automatically.">
                <input
                  className="input"
                  placeholder="192.168.1.42"
                  value={form.source_uri}
                  onChange={(e) => set('source_uri', e.target.value)}
                />
              </Field>
              <button className="btn-secondary" onClick={probeBoard} disabled={scanning}>
                {scanning ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : <Icon name="wifi_tethering" className="text-[16px]" />}
                Test
              </button>
            </div>
            {probe ? (
              <div
                className={`rounded-lg border p-space-sm flex flex-col gap-1 ${
                  probe.reachable ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'
                }`}
              >
                <p className="font-headline-sm text-headline-sm">
                  {probe.reachable
                    ? `Board reachable — will use ${probe.recommended_mode} mode`
                    : 'Board did not respond'}
                </p>
                {Object.entries(probe.endpoints).map(([name, endpoint]) => (
                  <p key={name} className="mono text-on-surface-variant truncate">
                    {endpoint.ok ? '✓' : '✗'} {name}: {endpoint.url}
                    {endpoint.error ? ` — ${endpoint.error}` : endpoint.status_code ? ` — HTTP ${endpoint.status_code}` : ''}
                  </p>
                ))}
                {probe.hint ? <p className="font-body-sm text-body-sm text-amber-900">{probe.hint}</p> : null}
              </div>
            ) : null}
          </div>
        ) : form.source_type === 'synthetic' ? (
          <p className="font-body-sm text-body-sm text-on-surface-variant bg-surface-container-low border-l-[3px] border-primary-container rounded-lg p-space-sm">
            A generated scene containing loitering, counter-flow, a following pair and a bag that is
            abandoned — useful for exercising every agent without hardware.
          </p>
        ) : (
          <Field
            label={form.source_type === 'file' ? 'Video file path' : 'Stream URL'}
            required
            hint={form.source_type === 'rtsp' ? 'Credentials below are injected into the URL when supplied.' : undefined}
          >
            <input
              className="input"
              placeholder={form.source_type === 'rtsp' ? 'rtsp://192.168.1.10:554/stream1' : 'http://…'}
              value={form.source_uri}
              onChange={(e) => set('source_uri', e.target.value)}
            />
          </Field>
        )}

        {/* identity + placement */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
          <Field label="Camera name" required>
            <input className="input" value={form.name} onChange={(e) => set('name', e.target.value)} placeholder="Gate 4 – North" />
          </Field>
          <Field label="Location">
            <input className="input" value={form.location} onChange={(e) => set('location', e.target.value)} placeholder="Terminal A, Concourse" />
          </Field>
          <Field label="Site">
            <select className="input" value={form.site_id} onChange={(e) => set('site_id', e.target.value)}>
              <option value="">Unassigned</option>
              {sites.map((site) => (
                <option key={site.id} value={site.id}>{site.name}</option>
              ))}
            </select>
          </Field>
          <Field label="Primary zone">
            <select className="input" value={form.zone_id} onChange={(e) => set('zone_id', e.target.value)}>
              <option value="">Unassigned</option>
              {zones.map((zone) => (
                <option key={zone.id} value={zone.id}>
                  {zone.name} ({zone.zone_type})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Processed on">
            <ProcessingSelect value={form.processing_node} onChange={(v) => set('processing_node', v)} />
          </Field>
        </div>

        <details className="rounded-lg border border-outline-variant/60">
          <summary className="px-space-md h-9 flex items-center cursor-pointer font-headline-sm text-headline-sm text-on-surface-variant">
            Capture, geometry and credentials
          </summary>
          <div className="p-space-md grid grid-cols-2 sm:grid-cols-4 gap-space-md border-t border-outline-variant/40">
            <Field label="Target FPS"><input type="number" className="input" value={form.fps} onChange={(e) => set('fps', Number(e.target.value))} /></Field>
            <Field label="Width"><input type="number" className="input" value={form.width} onChange={(e) => set('width', Number(e.target.value))} /></Field>
            <Field label="Height"><input type="number" className="input" value={form.height} onChange={(e) => set('height', Number(e.target.value))} /></Field>
            <Field label="Rotation">
              <select className="input" value={form.rotation} onChange={(e) => set('rotation', Number(e.target.value))}>
                {[0, 90, 180, 270].map((deg) => <option key={deg} value={deg}>{deg}°</option>)}
              </select>
            </Field>
            <Field label="Latitude"><input className="input" value={form.latitude} onChange={(e) => set('latitude', e.target.value)} placeholder="optional" /></Field>
            <Field label="Longitude"><input className="input" value={form.longitude} onChange={(e) => set('longitude', e.target.value)} placeholder="optional" /></Field>
            <Field label="Bearing °"><input type="number" className="input" value={form.orientation_deg} onChange={(e) => set('orientation_deg', Number(e.target.value))} /></Field>
            <Field label="Field of view °"><input type="number" className="input" value={form.field_of_view_deg} onChange={(e) => set('field_of_view_deg', Number(e.target.value))} /></Field>
            <Field label="Floor"><input type="number" className="input" value={form.floor} onChange={(e) => set('floor', Number(e.target.value))} /></Field>
            <Field label="Range (m)"><input type="number" className="input" value={form.range_m} onChange={(e) => set('range_m', Number(e.target.value))} /></Field>
            <Field label="Username"><input className="input" value={form.username} onChange={(e) => set('username', e.target.value)} placeholder="optional" /></Field>
            <Field label="Password"><input type="password" className="input" value={form.password} onChange={(e) => set('password', e.target.value)} placeholder="optional" /></Field>
          </div>
        </details>
      </div>
    </Modal>
  )
}

/* ---------------------------------------------------------------- detail */
function CameraDetail({
  camera,
  onClose,
  editable,
  onControl,
  onDelete,
  onSaved,
  zones,
}: {
  camera: Camera | null
  onClose: () => void
  editable: boolean
  onControl: (camera: Camera, action: 'start' | 'stop') => void
  onDelete: (camera: Camera) => void
  onSaved: () => void
  zones: { id: string; name: string; zone_type: string }[]
}) {
  const [redaction, setRedaction] = useState<boolean | null>(null)

  useEffect(() => {
    setRedaction(camera?.privacy_redaction ?? null)
  }, [camera])

  if (!camera) return null

  async function patch(body: Record<string, unknown>) {
    try {
      await api.patch(`/api/cameras/${camera!.id}`, body)
      toast('Camera updated')
      onSaved()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const runtime = camera.runtime
  return (
    <Modal
      open
      onClose={onClose}
      title={camera.name}
      width="max-w-4xl"
      footer={
        editable ? (
          <>
            <button className="btn-danger mr-auto" onClick={() => onDelete(camera)}>
              <Icon name="delete" className="text-[16px]" />
              Delete
            </button>
            {camera.running ? (
              <button className="btn-secondary" onClick={() => onControl(camera, 'stop')}>Stop</button>
            ) : (
              <button className="btn-primary" onClick={() => onControl(camera, 'start')}>Start</button>
            )}
          </>
        ) : null
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
        <div className="rounded-lg overflow-hidden border border-outline-variant/60 bg-slate-900 aspect-video">
          {camera.running ? (
            <img src={streamUrl(camera.id)} alt={`${camera.name} live view`} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-slate-400 mono">
              {camera.status_detail || 'not running'}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-space-md min-w-0">
          <dl className="grid grid-cols-2 gap-space-md">
            <Detail label="Status" value={titleCase(runtime?.status ?? camera.status)} />
            <Detail label="Source" value={`${camera.source_type} · ${camera.source_uri || 'n/a'}`} />
            <Detail label="Resolution" value={runtime ? `${runtime.resolution[0]}×${runtime.resolution[1]}` : `${camera.width}×${camera.height}`} />
            <Detail label="Measured FPS" value={runtime ? runtime.measured_fps.toFixed(1) : '—'} />
            <Detail label="Frames" value={runtime ? String(runtime.frame_index) : '—'} />
            <Detail label="Inference" value={runtime ? `${runtime.inference_ms.toFixed(0)} ms` : '—'} />
            <Detail label="Device" value={runtime?.device ?? '—'} />
            <Detail label="Detector" value={runtime?.detector?.backend ?? '—'} />
            <Detail
              label="Processed on"
              value={
                runtime?.processing_node
                  ? `${runtime.processing_node.name} · ${runtime.processing_node.gpu_name}`
                  : runtime
                    ? 'this server'
                    : '—'
              }
            />
          </dl>

          {camera.status_detail ? (
            <p className="font-body-sm text-body-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
              {camera.status_detail}
            </p>
          ) : null}

          {editable ? (
            <>
              <Field label="Processed on">
                <ProcessingSelect
                  value={String((camera.meta ?? {}).processing_node ?? '')}
                  onChange={(v) => {
                    const meta = { ...(camera.meta ?? {}) } as Record<string, unknown>
                    if (v) meta.processing_node = v
                    else delete meta.processing_node
                    patch({ meta })
                  }}
                />
              </Field>
              <Field label="Primary zone">
                <select
                  className="input"
                  value={camera.zone_id ?? ''}
                  onChange={(e) => patch({ zone_id: e.target.value || null })}
                >
                  <option value="">Unassigned</option>
                  {zones.map((zone) => (
                    <option key={zone.id} value={zone.id}>
                      {zone.name} ({zone.zone_type})
                    </option>
                  ))}
                </select>
              </Field>

              <Toggle
                checked={redaction ?? true}
                onChange={(next) => {
                  setRedaction(next)
                  patch({ privacy_redaction: next })
                }}
                label="Redact bystander faces"
                description="Blurs every face that is not the subject of an open incident, before the frame leaves the server."
              />
            </>
          ) : null}

          {runtime?.source ? (
            <details className="rounded-lg border border-outline-variant/60">
              <summary className="px-space-md h-8 flex items-center cursor-pointer font-headline-sm text-headline-sm text-on-surface-variant">
                Source diagnostics
              </summary>
              <pre className="p-space-md mono text-on-surface-variant overflow-x-auto border-t border-outline-variant/40">
                {JSON.stringify(runtime.source, null, 2)}
              </pre>
            </details>
          ) : null}
        </div>
      </div>
    </Modal>
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

function sourceIcon(value: string): string {
  return (
    {
      webcam: 'videocam',
      esp32cam: 'developer_board',
      rtsp: 'router',
      http_mjpeg: 'cloud',
      file: 'movie',
      synthetic: 'science',
    }[value] ?? 'lan'
  )
}

function sourceLabel(value: string): string {
  return (
    {
      webcam: 'External webcam',
      esp32cam: 'ESP32-CAM',
      rtsp: 'RTSP / IP camera',
      http_mjpeg: 'MJPEG stream',
      file: 'Video file',
      synthetic: 'Synthetic scene',
    }[value] ?? titleCase(value)
  )
}

/* ------------------------------------------------------- edge GPU nodes */
function ProcessedOn({ runtime }: { runtime: NonNullable<Camera['runtime']> }) {
  const node = runtime.processing_node
  if (!node) {
    return (
      <p className="mono text-outline truncate flex items-center gap-1">
        <Icon name="dns" className="text-[14px]" />
        this server · {runtime.device}
      </p>
    )
  }
  return (
    <p
      className={`mono truncate flex items-center gap-1 ${node.online ? 'text-emerald-700' : 'text-amber-700'}`}
      title={node.online ? 'Processed on an edge GPU node' : 'Edge node is not reporting'}
    >
      <Icon name="memory" className="text-[14px]" />
      {node.name} · {node.gpu_name}
      {node.online ? '' : ' (offline)'}
    </p>
  )
}

function ProcessingSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useApi<EdgeStatus>('/api/edge/nodes', undefined, { pollMs: 10000 })
  const nodes = data?.nodes ?? []
  return (
    <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{data ? `Default (${data.placement})` : 'Default'}</option>
      <option value="auto">Auto: a GPU node if one is online, else this server</option>
      <option value="server">This server</option>
      <option value="edge">Any edge GPU node (never this server)</option>
      {nodes.map((n) => (
        <option key={n.node_id} value={n.node_id} disabled={!n.online}>
          {n.name} · {n.gpu_name}
          {n.online ? '' : ' (offline)'}
        </option>
      ))}
    </select>
  )
}
