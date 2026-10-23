from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class UserRole(str):
    ATTENDING_PHYSICIAN = "attending_physician"
    STAFF_NURSE = "staff_nurse"
    HOSPITAL_ADMIN = "hospital_admin"
    CLINICAL_INTERN = "clinical_intern"

class DocumentMetadata(BaseModel):
    id: str
    title: str
    code: str
    department: str
    version: str
    effective_date: str
    status: str  # "active", "superseded", "draft"
    access_level: str  # "GENERAL", "CLINICAL", "RESTRICTED_ICU", "ADMIN"
    file_type: str = "txt"
    pages_count: int = 1
    description: str = ""

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_code: str
    doc_title: str
    version: str
    effective_date: str
    access_level: str
    page_number: int
    section: str
    content: str

class Citation(BaseModel):
    doc_code: str
    doc_title: str
    version: str
    page_number: int
    section: str
    passage_excerpt: str
    confidence: float
    is_superseded: bool = False

class ChatRequest(BaseModel):
    query: str
    user_role: str = "attending_physician"
    encounter_id: Optional[str] = None
    department_filter: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    conflict_warning: Optional[str] = None
    abstention_reason: Optional[str] = None
    access_blocked: bool = False
    model_mode: str = "RAG-Grounded"
    latency_ms: float = 0.0

class Encounter(BaseModel):
    id: str
    title: str
    time: str
    category: str
    patient_name: str
    patient_id: str
    duration: str
    subjective: str
    objective: str
    assessment: str
    plan: str
    patient_handout: str
    sick_note: str

class AuditEntry(BaseModel):
    timestamp: str
    user_role: str
    query: str
    status: str  # "ALLOWED", "BLOCKED", "ABSTAINED"
    retrieved_count: int
    documents_accessed: List[str]
    latency_ms: float

class EvalMetricResult(BaseModel):
    category: str
    total_tests: int
    passed_tests: int
    recall_at_5: float
    citation_correctness: float
    answer_correctness: float
    unsupported_rate: float
    abstention_f1: float
    access_violations: int
    avg_latency_ms: float

# ==================== AUTHENTICATION MODELS ====================
class LoginRequest(BaseModel):
    username_or_email: str
    password: str
    workstation_id: Optional[str] = "WS-CLINIC-401"

class AuthUser(BaseModel):
    user_id: str
    name: str
    email: str
    role_id: str
    role_label: str
    license_number: str
    department: str
    avatar_initials: str
    clearance_badge: str
    permissions: List[str]

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    user: AuthUser
    message: str = "Authentication successful"

class SessionStatus(BaseModel):
    authenticated: bool
    user: Optional[AuthUser] = None

