# 🔐 Authentication System - RSJD Audio Classifier

## 📋 Overview

Sistem authentication untuk aplikasi Streamlit **Klasifikasi Gangguan Jiwa Berdasarkan Audio** menggunakan SQLite database dengan password hashing PBKDF2-SHA256.

---

## 🚀 Quick Start

### 1. Setup Database & Initial Users

```bash
# Run setup script (ONLY ONCE!)
python setup_auth.py
```

This will create:
- `auth/users.db` - SQLite database
- 3 initial users (admin, doctor, staff)

### 2. Login to Application

```bash
streamlit run app_optimized.py
```

**Default Credentials:**

| Username | Password | Role |
|----------|----------|------|
| `admin` | `Admin123` | Administrator |
| `dr_amino` | `Doctor123` | Doctor |
| `staff1` | `Staff123` | Staff |

⚠️ **IMPORTANT:** Change all default passwords immediately after first login!

---

## 👥 User Roles

### **1. Administrator (admin)**
- ✅ Full system access
- ✅ Create/deactivate users
- ✅ View all login history
- ✅ Database backup/restore
- ✅ Run audio analysis

### **2. Doctor (doctor)**
- ✅ Run audio analysis
- ✅ View own history
- ✅ Export reports
- ❌ Cannot manage users

### **3. Staff (staff)**
- ✅ Run audio analysis
- ✅ View results
- ❌ Cannot manage users
- ❌ Limited history access

---

## 🔒 Security Features

### **Password Security:**
- **Algorithm:** PBKDF2-SHA256
- **Iterations:** 100,000
- **Salt:** 32-character random hex (per password)
- **Storage:** `salt$hash` format

### **Password Requirements:**
- ✅ Minimum 8 characters
- ✅ At least 1 uppercase letter
- ✅ At least 1 lowercase letter
- ✅ At least 1 number
- ❌ Cannot use common weak passwords

### **Session Management:**
- **Timeout:** 30 minutes of inactivity
- **Auto-logout:** After timeout period
- **Session state:** Streamlit session_state

### **Audit Logging:**
- All login attempts logged ( success/failed)
- Username, timestamp, IP address recorded
- Accessible via Admin Panel

---

## 📂 File Structure

```
F:\Classifier_v2\
├── auth/
│   ├── __init__.py           # Package initialization
│   ├── authenticator.py      # Main authentication logic
│   ├── password_utils.py     # Password hashing/verification
│   ├── database.py           # SQLite operations
│   └── users.db              # 🔐 USER CREDENTIALS (SECURE!)
│
├── config/
│   └── (reserved for future config files)
│
├── logs/
│   └── (application logs)
│
├── backups/
│   └── users_backup_*.db     # Database backups
│
├── setup_auth.py             # One-time setup script
├── admin_panel.py            # Admin user management UI
└── app_optimized.py          # Main Streamlit app (with auth)
```

---

## 🛠️ Admin Panel

### Access Admin Panel:

```bash
streamlit run admin_panel.py
```

Login with **admin** credentials.

### Features:

#### **📊 Dashboard**
- Total users statistics
- Active/inactive users count
- Users by role breakdown
- Recent login activity

#### **👥 Manage Users**
- View all users in table
- Activate/deactivate users
- Change user passwords
- View user details

#### **➕ Add User**
- Create new users
- Set username, full name, role
- Generate secure password
- Instant activation

#### **📜 Login History**
- View all login attempts
- Filter by status (success/failed)
- Filter by username
- Export to CSV
- Database backup

---

## 🔧 Common Tasks

### **Add New User:**

**Option 1: Via Admin Panel (Recommended)**
```bash
streamlit run admin_panel.py
# Navigate to "Add User" tab
# Fill form and submit
```

**Option 2: Via Python Script**
```python
from auth import create_user, hash_password

password_hash = hash_password("SecurePassword123")
create_user(
    username="new_doctor",
    password_hash=password_hash,
    full_name="Dr. New Doctor",
    role="doctor",
    created_by="admin"
)
```

### **Change Password:**

**Option 1: Via Admin Panel**
```bash
# Login to admin_panel.py
# Go to "Manage Users" → Select user → Change Password
```

**Option 2: Via Code**
```python
from auth import change_password, hash_password

new_hash = hash_password("NewPassword123")
change_password("username", new_hash)
```

### **Deactivate User:**

```python
from auth import deactivate_user

deactivate_user("username")  # Soft delete
```

### **Activate User:**

```python
from auth import activate_user

activate_user("username")
```

### ** Backup Database:**

**Option 1: Via Admin Panel**
```bash
# Login to ad min_panel.py
# Go to "Login History" → Click "Backup Database"
```

**Option 2: Via Code**
```python
from auth import backup_database

backup_database()  # Creates timestamped backup
```

---

## 📊 Database Schema

### **Table: users**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `username` | TEXT | Unique username |
| `password_hash` | TEXT | Hashed password (salt$hash) |
| `full_name` | TEXT | Full name |
| `role` | TEXT | User role (admin/doctor/staff) |
| `created_at` | TIMESTAMP | Creation timestamp |
| `last_login` | TIMESTAMP | Last login timestamp |
| `is_active` | BOOLEAN | Active status (1=active, 0=inactive) |
| `created_by` | TEXT | Creator username |

### **Table: login_history**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `user_id` | INTEGER | Foreign key to users.id |
| `username` | TEXT | Username that attempted login |
| `login_time` | TIMESTAMP | Login attempt timestamp |
| `success` | BOOLEAN | Success status (1=success, 0=failed) |
| `ip_address` | TEXT | IP address |

---

## 🔍 Troubleshooting

### **Problem: "Username already exists"**
**Solution:** Choose different username or deactivate existing user

### **Problem: "Password too weak"**
**Solution:** Ensure password meets all requirements:
- Min 8 characters
- 1 uppercase, 1 lowercase, 1 number

### **Problem: "Database locked"**
**Solution:** Close all connections to database:
```python
# Close any open database connections
# Or restart the app
```

### **Problem: "Session timeout"**
**Solution:** Login again (sessions expire after 30 minutes)

### **Problem: "Forgot admin password"**
**Solution:** Reset via setup script:
```bash
# Delete users.db
# Run: python setup_auth.py
# Use default admin credentials
```

---

## 📁 Backup & Recovery

### **Manual Backup:**

```bash
# Copy database file
copy auth\users.db backups\users_backup_YYYYMMDD.db
```

### **Automated Backup:**

Add to scheduled task (Windows):
```powershell
# Run daily at 2 AM
schtasks /create /tn "RSJD_DB_Backup" /tr "python F:\Classifier_v2\backup_script.py" /sc daily /st 02:00
```

### **Restore from Backup:**

```bash
# Stop Streamlit app first!
# Then copy backup over current database
copy backups\users_backup_YYYYMMDD.db auth\users.db
```

---

## 🔐 Security Best Practices

### **✅ DO:**
- Change default passwords IMMEDIATELY
- Use strong passwords (12+ characters)
- Regularly backup database
- Review login history for suspicious activity
- Deactivate users when they leave
- Keep `auth/users.db` secure (restrict file permissions)

### **❌ DON'T:**
- Share admin credentials
- Use common passwords
- Commit `users.db` to git
- Leave default passwords
- Give everyone admin access
- Expose database publicly

---

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review code comments in `auth/` directory
3. Contact system administrator

---

## 📝 Changelog

### **v1.0.0** (2025-11-28)
- ✅ Initial authentication system
- ✅ SQLite database with PBKDF2-SHA256
- ✅ Role-based access control (admin/doctor/staff)
- ✅ Session management with 30min timeout
- ✅ Complete admin panel
- ✅ Login history & audit logging
- ✅ Database backup functionality

---

**© 2025 RSJD dr. Amino Gondohutomo**  
**Secured by Deep Learning & Modern Authentication** 🔒
