# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""One clinical headline contract shared by the interface, PDF, history and batch."""

def transfer_closed(patient):
    """No transfer is possible in the current cycle given the entered results.

    Without PGT-A every blastocyst is transferable, so zero good-quality
    blastocysts alone do not close the cycle. With PGT-A (an entered euploid
    count) zero euploid embryos do.
    """
    patient=patient or {}
    if any(patient.get('known_'+key)==0 for key in ('okk','mii','pn2','blasts')):return True
    return patient.get('known_euploid')==0


def clinical_summary(result):
    patient=result.get('patient') or {}
    confirmed=transfer_closed(patient)
    return {'probability':0.0 if confirmed else result.get('headline'),
        'kind':'current_cycle' if confirmed else 'per_transfer',
        'no_transfer_confirmed':confirmed}


def csdi_note(assessment,language='ru'):
    reason=(assessment or {}).get('reason')
    notes={
        'no_2pn':('CSDI не запускается при отсутствии 2PN.','CSDI is not run without 2PN.'),
        'model_unavailable':('Модель CSDI недоступна и не участвует в объединении.','The CSDI model is unavailable and does not enter fusion.'),
        'inference_failed':('Расчёт CSDI не завершён; его оценка не участвует в объединении.','CSDI did not complete; its estimate does not enter fusion.'),
        'unchecked':('Область применимости CSDI не проверена; оценка исключена из объединения.','CSDI applicability was not assessed; its estimate is excluded from fusion.'),
        'outside_training':('Оценка CSDI исключена из объединения из-за выхода за область обучающих данных.','The CSDI estimate is excluded from fusion because the case is outside its training domain.')}
    return notes.get(reason,('CSDI не участвует в объединённом прогнозе.','CSDI does not enter the combined forecast.'))[0 if language=='ru' else 1]
