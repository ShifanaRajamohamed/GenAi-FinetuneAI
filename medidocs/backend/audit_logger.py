import time
from datetime import datetime
from typing import List
from .models import AuditEntry

class AuditLogger:
    def __init__(self):
        self.logs: List[AuditEntry] = []
        # Prepopulate with sample realistic clinical audit trail
        self._init_mock_logs()

    def _init_mock_logs(self):
        sample_events = [
            ("attending_physician", "Check medication storage temperature excursion protocol", "ALLOWED", 3, ["POL-PHARM-045"], 182.4),
            ("staff_nurse", "Adult sepsis screening qSOFA bedside threshold", "ALLOWED", 2, ["SOP-CLIN-012"], 145.1),
            ("clinical_intern", "Mechanical ventilation weaning protocol SOP-ICU-088", "BLOCKED", 0, [], 18.2),
            ("staff_nurse", "Airborne PPE isolation negative pressure specs", "ALLOWED", 2, ["POL-INFECT-009"], 190.5),
            ("clinical_intern", "What is the Wi-Fi password for visitor lounge?", "ABSTAINED", 0, [], 92.0),
            ("hospital_admin", "ED staff fatigue consecutive hours policy ADM-HR-101", "ALLOWED", 1, ["ADM-HR-101"], 134.8),
            ("staff_nurse", "Export confidential RSBI formulas for ICU extubation", "BLOCKED", 0, [], 14.9),
            ("attending_physician", "Code blue attending physician co-signature timeline", "ALLOWED", 1, ["SOP-CODE-BLUE-004"], 167.3)
        ]
        base_time = time.time() - 3600 * 4
        for idx, (role, q, status, count, docs, lat) in enumerate(sample_events):
            t_str = datetime.fromtimestamp(base_time + idx * 1200).strftime("%Y-%m-%d %H:%M:%S")
            self.logs.append(AuditEntry(
                timestamp=t_str,
                user_role=role,
                query=q,
                status=status,
                retrieved_count=count,
                documents_accessed=docs,
                latency_ms=lat
            ))

    def log(self, user_role: str, query: str, status: str, retrieved_count: int, docs: List[str], latency_ms: float):
        entry = AuditEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_role=user_role,
            query=query,
            status=status,
            retrieved_count=retrieved_count,
            documents_accessed=docs,
            latency_ms=round(latency_ms, 1)
        )
        self.logs.insert(0, entry)
        if len(self.logs) > 200:
            self.logs.pop()

    def get_logs(self, limit: int = 50) -> List[AuditEntry]:
        return self.logs[:limit]
