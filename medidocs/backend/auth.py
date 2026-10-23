import time
import secrets
import hashlib
from typing import Dict, Optional, List
from .models import AuthUser, LoginRequest, LoginResponse
from .rag_engine import ROLE_PERMISSIONS, ROLE_LABELS

# Mock Clinical Users Database
USERS_DB: Dict[str, Dict] = {
    "dr.chen@hospital.org": {
        "user_id": "usr-physician-01",
        "username": "chen_md",
        "email": "dr.chen@hospital.org",
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "name": "Dr. Sarah Chen, MD",
        "role_id": "attending_physician",
        "license_number": "MD-884920",
        "department": "Internal Medicine & Pulmonary",
        "avatar_initials": "SC",
        "clearance_badge": "CLINICAL + RESTRICTED_ICU",
    },
    "nurse.vance@hospital.org": {
        "user_id": "usr-nurse-02",
        "username": "vance_rn",
        "email": "nurse.vance@hospital.org",
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "name": "Nurse Marcus Vance, RN",
        "role_id": "staff_nurse",
        "license_number": "RN-551029",
        "department": "Emergency & Critical Care Nursing",
        "avatar_initials": "MV",
        "clearance_badge": "CLINICAL (General + Infection Control)",
    },
    "admin.rostova@hospital.org": {
        "user_id": "usr-admin-03",
        "username": "rostova_admin",
        "email": "admin.rostova@hospital.org",
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "name": "Elena Rostova",
        "role_id": "hospital_admin",
        "license_number": "ADM-1002",
        "department": "Clinical Governance & Hospital Operations",
        "avatar_initials": "ER",
        "clearance_badge": "ADMIN + AUDIT DISCLOSURES",
    },
    "intern.miller@hospital.org": {
        "user_id": "usr-intern-04",
        "username": "miller_intern",
        "email": "intern.miller@hospital.org",
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "name": "David Miller",
        "role_id": "clinical_intern",
        "license_number": "INT-449102",
        "department": "Internal Medicine Residency (Year 1)",
        "avatar_initials": "DM",
        "clearance_badge": "GENERAL STAFF CLEARANCE",
    }
}

# Active Session Token Store: token -> AuthUser
ACTIVE_SESSIONS: Dict[str, AuthUser] = {}

class AuthService:
    def __init__(self):
        pass

    def authenticate(self, req: LoginRequest) -> Optional[LoginResponse]:
        identifier = req.username_or_email.lower().strip()
        pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()

        # Find user by email or username
        user_record = None
        for record in USERS_DB.values():
            if record["email"].lower() == identifier or record["username"].lower() == identifier:
                user_record = record
                break

        if not user_record:
            return None

        # Check password hash (accept 'password123' or direct match)
        if user_record["password_hash"] != pwd_hash and req.password != "password123":
            return None

        role_id = user_record["role_id"]
        permissions = list(ROLE_PERMISSIONS.get(role_id, []))
        role_label = ROLE_LABELS.get(role_id, role_id)

        auth_user = AuthUser(
            user_id=user_record["user_id"],
            name=user_record["name"],
            email=user_record["email"],
            role_id=role_id,
            role_label=role_label,
            license_number=user_record["license_number"],
            department=user_record["department"],
            avatar_initials=user_record["avatar_initials"],
            clearance_badge=user_record["clearance_badge"],
            permissions=permissions
        )

        token = f"medidocs_token_{secrets.token_hex(16)}"
        ACTIVE_SESSIONS[token] = auth_user

        return LoginResponse(
            access_token=token,
            token_type="Bearer",
            user=auth_user,
            message="Hospital identity verified. Clearance session established."
        )

    def validate_token(self, token: str) -> Optional[AuthUser]:
        if not token:
            return None
        # Clean Bearer prefix if provided
        clean_token = token.replace("Bearer ", "").strip()
        return ACTIVE_SESSIONS.get(clean_token)

    def logout(self, token: str) -> bool:
        clean_token = token.replace("Bearer ", "").strip()
        if clean_token in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[clean_token]
            return True
        return False

    def get_all_demo_users(self) -> List[Dict]:
        return [
            {
                "email": u["email"],
                "username": u["username"],
                "name": u["name"],
                "role_id": u["role_id"],
                "role_label": ROLE_LABELS.get(u["role_id"], u["role_id"]),
                "license_number": u["license_number"],
                "department": u["department"],
                "clearance_badge": u["clearance_badge"],
                "avatar_initials": u["avatar_initials"]
            }
            for u in USERS_DB.values()
        ]
