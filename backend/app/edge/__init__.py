"""Edge GPU processing: run cameras on operators' own GPUs, aggregate here.

hub.py   - the server side (the VPS): node registry, camera placement,
           RemoteCameraWorker proxies, ingest of what nodes report.
node.py  - the edge side: runs cameras on the local GPU with the unchanged
           pipeline and pushes results up to the server.
"""
