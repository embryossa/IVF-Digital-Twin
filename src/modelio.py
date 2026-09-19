# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""One loader for developer plaintext and authenticated clinic model payloads."""
import os
import hashlib
import json
from pathlib import Path

_VERIFIED={}

def verify_model_integrity(path):
    """Detect damaged review payloads before deserialisation.

    The unsigned review manifest is an integrity check, not an authenticity
    guarantee. Signing and anti-tamper protection are explicitly deferred.
    """
    root=Path(__file__).resolve().parents[1]
    manifest=root/'review-manifest.json'
    if not manifest.is_file():return
    target=Path(path).resolve()
    relative=target.relative_to(root).as_posix()
    stat=target.stat();ms=manifest.stat()
    key=(str(target),stat.st_mtime_ns,stat.st_size,ms.st_mtime_ns,ms.st_size)
    if key in _VERIFIED:return
    rows=json.loads(manifest.read_text(encoding='utf-8'))['files']
    expected=next((row['sha256'] for row in rows if row['path']==relative),None)
    if expected is None:raise ValueError('Model is absent from the review manifest')
    with target.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual != expected:raise ValueError('Model integrity check failed: '+relative)
    _VERIFIED[key]=True

def model_exists(path):
    return os.path.exists(path) or os.path.isfile(str(path)+'.enc')

def model_source(path):
    encrypted=str(path)+'.enc'
    if os.path.isfile(encrypted):
        verify_model_integrity(encrypted)
        from crypt_engine import decrypt_model_to_stream
        return decrypt_model_to_stream(encrypted)
    verify_model_integrity(path)
    return path

def load_torch(path,*args,**kwargs):
    import torch
    return torch.load(model_source(path),*args,**kwargs)

def load_joblib(path,*args,**kwargs):
    import joblib
    return joblib.load(model_source(path),*args,**kwargs)
