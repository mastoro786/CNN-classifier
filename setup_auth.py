"""
Setup Initial Users & Database
Run this script ONCE untuk create database dan admin user pertama
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from auth.database import init_database, create_user
from auth.password_utils import hash_password


def setup_database():
    """Initialize database"""
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    print()
    
    init_database()
    print()


def create_initial_users():
    """Create initial admin and sample users"""
    print("=" * 60)
    print("CREATING INITIAL USERS")
    print("=" * 60)
    print()
    
    users_to_create = [
        {
            'username': 'admin',
            'password': 'Admin123',  # CHANGE THIS!
            'full_name': 'Administrator',
            'role': 'admin'
        },
        {
            'username': 'dr_amino',
            'password': 'Doctor123',  # CHANGE THIS!
            'full_name': 'Dr. Amino Gondohutomo',
            'role': 'doctor'
        },
        {
            'username': 'staff1',
            'password': 'Staff123',  # CHANGE THIS!
            'full_name': 'Staff RSJD',
            'role': 'staff'
        }
    ]
    
    for user_data in users_to_create:
        password_hash = hash_password(user_data['password'])
        
        success = create_user(
            username=user_data['username'],
            password_hash=password_hash,
            full_name=user_data['full_name'],
            role=user_data['role'],
            created_by='setup_script'
        )
        
        if success:
            print(f"   Username: {user_data['username']}")
            print(f"   Password: {user_data['password']}")
            print(f"   Role: {user_data['role']}")
            print()


def main():
    """Main setup function"""
    print()
    print("🚀 RSJD Classifier - Authentication Setup")
    print()
    
    # Step 1: Initialize database
    setup_database()
    
    # Step 2: Create initial users
    create_initial_users()
    
    print("=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT SECURITY NOTES:")
    print("   1. CHANGE ALL DEFAULT PASSWORDS IMMEDIATELY!")
    print("   2. Delete this script after setup OR")
    print("   3. Remove password information from this file")
    print()
    print("Default credentials created:")
    print("   • admin / Admin123 (Administrator)")
    print("   • dr_amino / Doctor123 (Doctor)")
    print("   • staff1 / Staff123 (Staff)")
    print()
    print("📍 Database location: F:\\Classifier_v2\\auth\\users.db")
    print()
    print("Next steps:")
    print("   1. Run: streamlit run app_optimized.py")
    print("   2. Login dengan credentials di atas")
    print("   3. Change password dari UI")
    print("   4. Add more users as needed")
    print()


if __name__ == "__main__":
    main()
