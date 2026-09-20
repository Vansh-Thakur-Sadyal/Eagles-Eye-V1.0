/** Data-fetching, live-stream and GPU hooks shared by every screen. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api, wsUrl, type GpuStatus } from './api'

/** Fetch once, with optional polling and a manual refresh handle. */
export function useApi<T>(
  path: string | null,
  params?: Record<string, unknown>,
  options: { pollMs?: number; enabled?: boolean } = {},
) {
  const { pollMs, enabled = true } = options
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(Boolean(path) && enabled)
  const key = JSON.stringify(params ?? {})
  const mounted = useRef(true)

  const load = useCallback(
    async (quiet = false) => {
      if (!path || !enabled) return
      if (!quiet) setLoading(true)
      try {
        const result = await api.get<T>(path, params)
        if (!mounted.current) return
        setData(result)
        setError(null)
      } catch (err) {
        if (!mounted.current) return
        setError(err instanceof ApiError ? err.message : String(err))
      } finally {
        if (mounted.current) setLoading(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [path, key, enabled],
  )

  useEffect(() => {
    mounted.current = true
    load()
    return () => {
      mounted.current = false
    }
  }, [load])

  useEffect(() => {
    if (!pollMs || !path || !enabled) return
    const timer = setInterval(() => load(true), pollMs)
    return () => clearInterval(timer)
  }, [pollMs, path, enabled, load])

  return { data, error, loading, refresh: () => load(true), reload: () => load(false) }
}

export interface StreamEvent {
  topic: string
  payload: Record<string, any>
  at: string
  source: string
  severity: string
}

/**
 * One shared WebSocket for the whole app.
 *
 * Reconnects with backoff and replays nothing on its own; screens that need
 * history call the REST endpoints. Keeping a single socket avoids N sockets
 * for N mounted panels.
 */
let socket: WebSocket | null = null
let listeners = new Set<(event: StreamEvent) => void>()
let connectionListeners = new Set<(connected: boolean) => void>()
let retry = 0
let reconnectTimer: number | undefined

function ensureSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return
  }
  try {
    socket = new WebSocket(wsUrl())
  } catch {
    return
  }
  socket.onopen = () => {
    retry = 0
    connectionListeners.forEach((fn) => fn(true))
  }
  socket.onmessage = (raw) => {
    try {
      const event = JSON.parse(raw.data) as StreamEvent
      listeners.forEach((fn) => fn(event))
    } catch {
      /* ignore malformed frames */
    }
  }
  socket.onclose = () => {
    connectionListeners.forEach((fn) => fn(false))
    socket = null
    if (listeners.size === 0) return
    retry += 1
    const delay = Math.min(15000, 500 * 2 ** Math.min(retry, 5))
    window.clearTimeout(reconnectTimer)
    reconnectTimer = window.setTimeout(ensureSocket, delay)
  }
  socket.onerror = () => socket?.close()
}

export function useLiveEvents(
  topics: string[],
  handler: (event: StreamEvent) => void,
  deps: unknown[] = [],
) {
  const stable = useRef(handler)
  stable.current = handler

  useEffect(() => {
    const listener = (event: StreamEvent) => {
      if (topics.length === 0 || topics.some((t) => event.topic.startsWith(t))) {
        stable.current(event)
      }
    }
    listeners.add(listener)
    ensureSocket()
    return () => {
      listeners.delete(listener)
      if (listeners.size === 0) {
        window.clearTimeout(reconnectTimer)
        socket?.close()
        socket = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topics.join(','), ...deps])
}

export function useConnection(): boolean {
  const [connected, setConnected] = useState(socket?.readyState === WebSocket.OPEN)
  useEffect(() => {
    connectionListeners.add(setConnected)
    ensureSocket()
    return () => {
      connectionListeners.delete(setConnected)
    }
  }, [])
  return connected
}

/**
 * GPU state, shared between the header toggle and the Settings screen.
 *
 * `serverGpu` controls where torch runs on the API host. `clientGpu` controls
 * whether the browser uses its high-performance WebGPU adapter for the twin and
 * heatmap rendering - that needs no permission prompt, which is why the
 * dashboard can use a viewer's discrete GPU when hosted on a CPU-only VPS.
 */
const CLIENT_GPU_KEY = 'sentinel.clientGpu'

export function useGpu() {
  const [status, setStatus] = useState<GpuStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clientGpu, setClientGpuState] = useState<boolean>(() => {
    try {
      return localStorage.getItem(CLIENT_GPU_KEY) !== 'off'
    } catch {
      return true
    }
  })

  const load = useCallback(async () => {
    try {
      setStatus(await api.get<GpuStatus>('/api/system/gpu'))
      setError(null)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [load])

  useLiveEvents(['system.gpu'], () => load(), [])

  const setServerGpu = useCallback(async (enabled: boolean) => {
    setBusy(true)
    try {
      const next = await api.post<GpuStatus>('/api/system/gpu', { enabled })
      setStatus(next)
      setError(next.warning ?? null)
      return next
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
      throw err
    } finally {
      setBusy(false)
    }
  }, [])

  const setClientGpu = useCallback((enabled: boolean) => {
    setClientGpuState(enabled)
    try {
      localStorage.setItem(CLIENT_GPU_KEY, enabled ? 'on' : 'off')
    } catch {
      /* ignore */
    }
  }, [])

  return { status, busy, error, refresh: load, setServerGpu, clientGpu, setClientGpu }
}

/** Detect the browser's WebGPU adapter without prompting the user. */
export function useWebGpu(enabled: boolean) {
  const [info, setInfo] = useState<{ available: boolean; adapter?: string; reason?: string }>({
    available: false,
  })

  useEffect(() => {
    let cancelled = false
    async function probe() {
      const nav = navigator as Navigator & { gpu?: any }
      if (!nav.gpu) {
        if (!cancelled) setInfo({ available: false, reason: 'This browser does not expose WebGPU.' })
        return
      }
      if (!enabled) {
        if (!cancelled) setInfo({ available: false, reason: 'Client acceleration is switched off.' })
        return
      }
      try {
        // high-performance selects a discrete GPU where one exists, with no prompt.
        const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' })
        if (cancelled) return
        if (!adapter) {
          setInfo({ available: false, reason: 'No WebGPU adapter was offered.' })
          return
        }
        const described = (await adapter.requestAdapterInfo?.()) ?? {}
        setInfo({
          available: true,
          adapter: [described.vendor, described.architecture, described.description]
            .filter(Boolean)
            .join(' ') || 'WebGPU adapter',
        })
      } catch (err) {
        if (!cancelled) setInfo({ available: false, reason: String(err) })
      }
    }
    probe()
    return () => {
      cancelled = true
    }
  }, [enabled])

  return info
}

/** Debounce any rapidly-changing value (search boxes, sliders). */
export function useDebounced<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return debounced
}
