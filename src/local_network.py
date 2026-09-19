# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""Keep local model clients local, including standalone script usage."""
import ipaddress
from urllib.parse import urlsplit

def local_ollama_host(value):
    value=str(value).strip()
    parsed=urlsplit(value if '://' in value else 'http://'+value)
    host=parsed.hostname or ''
    try:
        valid=host=='localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:valid=False
    if (not valid or parsed.scheme not in ('http','https') or parsed.username or
            parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment):
        raise ValueError('OLLAMA_HOST must be a loopback URL')
    port=11434 if parsed.port is None else parsed.port
    if not 1<=port<=65535:raise ValueError('Invalid local port')
    if host=='localhost':host='127.0.0.1'
    if ':' in host:host='['+host+']'
    return f'{parsed.scheme}://{host}:{port}'
