"""
Authentication Manager for Streamlit App
"""

import streamlit as st
from typing import Optional, Dict
from datetime import datetime, timedelta

from auth.password_utils import verify_password, hash_password, validate_password_strength
from auth.database import (
    get_user, update_last_login, log_login_attempt,
    get_all_users, create_user, deactivate_user, activate_user, change_password
)


class Authenticator:
    """Handle authentication untuk Streamlit app"""
    
    def __init__(self):
        """Initialize authenticator"""
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state['authenticated'] = False
        if 'user' not in st.session_state:
            st.session_state['user'] = None
        if 'login_time' not in st.session_state:
            st.session_state['login_time'] = None
    
    def login(self, username: str, password: str) -> tuple[bool, str]:
        """
        Login user
        
        Args:
            username: Username
            password: Password (plain text)
            
        Returns:
            (success, message)
        """
        # Get user from database
        user = get_user(username)
        
        if not user:
            log_login_attempt(username, False)
            return False, "Username atau password salah"
        
        # Verify password
        if not verify_password(user['password_hash'], password):
            log_login_attempt(username, False)
            return False, "Username atau password salah"
        
        # Check if user is active
        if not user['is_active']:
            log_login_attempt(username, False, user['id'])
            return False, "Akun Anda telah dinonaktifkan. Hubungi administrator."
        
        # Login successful
        st.session_state['authenticated'] = True
        st.session_state['user'] = {
            'id': user['id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'role': user['role']
        }
        st.session_state['login_time'] = datetime.now()
        
        # Update last login
        update_last_login(username)
        log_login_attempt(username, True, user['id'])
        
        return True, f"Selamat datang, {user['full_name']}!"
    
    def logout(self):
        """Logout user"""
        st.session_state['authenticated'] = False
        st.session_state['user'] = None
        st.session_state['login_time'] = None
        st.rerun()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        if not st.session_state.get('authenticated', False):
            return False
        
        # Check session timeout (30 minutes)
        if st.session_state.get('login_time'):
            elapsed = datetime.now() - st.session_state['login_time']
            if elapsed > timedelta(minutes=30):
                self.logout()
                return False
        
        return True
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current logged in user"""
        return st.session_state.get('user')
    
    def has_role(self, required_role: str) -> bool:
        """
        Check if current user has required role
        
        Args:
            required_role: Required role (admin, doctor, staff)
            
        Returns:
            True if user has role
        """
        user = self.get_current_user()
        if not user:
            return False
        
        # Admin has all roles
        if user['role'] == 'admin':
            return True
        
        return user['role'] == required_role
    
    def require_auth(self, show_login_page_func):
        """
        Decorator-like function to require authentication
        
        Usage:
            auth = Authenticator()
            if not auth.require_auth(show_login_page):
                return  # Stop execution
            # Continue with authenticated code
        """
        if not self.is_authenticated():
            show_login_page_func()
            return False
        return True
    
    def show_login_page(self):
        """Show login page"""
        # Custom CSS untuk login page
        st.markdown("""
            <style>
            .login-header {
                text-align: center;
                padding: 2rem 0;
                background: linear-gradient(135deg, #5B3A8C 0%, #3D2564 100%);
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .login-header h1 {
                color: white;
                margin: 0;
            }
            .login-header p {
                color: #e0e0e0;
                margin-top: 0.5rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
            <div class="login-header">
                <h1>🎙️ Klasifikasi Gangguan Jiwa</h1>
                <p>RSJD dr. Amino Gondohutomo</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Login form
        st.markdown("### 🔐 Login")
        
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("👤 Username", key="login_username")
            password = st.text_input("🔒 Password", type="password", key="login_password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
            with col2:
                st.form_submit_button("❌ Clear", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Username dan password harus diisi!")
                else:
                    with st.spinner("🔄 Memverifikasi credentials..."):
                        success, message = self.login(username, password)
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #888; font-size: 0.9rem;">
                <p>© 2025 RSJD dr. Amino Gondohutomo</p>
                <p>Powered by Deep Learning & Streamlit</p>
            </div>
        """, unsafe_allow_html=True)
    
    def show_user_info_sidebar(self):
        """Show user info in sidebar"""
        if not self.is_authenticated():
            return
        
        user = self.get_current_user()
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Information")
            st.markdown(f"**Nama:** {user['full_name']}")
            st.markdown(f"**Username:** {user['username']}")
            st.markdown(f"**Role:** {user['role'].upper()}")
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                self.logout()


# Singleton instance
_authenticator = None

def get_authenticator() -> Authenticator:
    """Get singleton authenticator instance"""
    global _authenticator
    if _authenticator is None:
        _authenticator = Authenticator()
    return _authenticator
