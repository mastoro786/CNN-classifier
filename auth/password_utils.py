"""
Password Hashing & Verification Utilities
Menggunakan PBKDF2-SHA256 untuk security
"""

import hashlib
import secrets
import re


def hash_password(password: str) -> str:
    """
    Hash password menggunakan PBKDF2-SHA256 dengan salt
    
    Args:
        password: Plain text password
        
    Returns:
        String format: "salt$hash"
    """
    # Generate random salt (32 characters hex)
    salt = secrets.token_hex(16)
    
    # Hash password dengan PBKDF2
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 100,000 iterations untuk security
    )
    
    # Return dalam format: salt$hash
    return f"{salt}${pwd_hash.hex()}"


def verify_password(stored_hash: str, provided_password: str) -> bool:
    """
    Verify password dengan stored hash
    
    Args:
        stored_hash: Hash dari database (format: salt$hash)
        provided_password: Password yang diinput user
        
    Returns:
        True jika password cocok, False jika tidak
    """
    try:
        # Split salt dan hash
        salt, pwd_hash = stored_hash.split('$')
        
        # Hash provided password dengan salt yang sama
        new_hash = hashlib.pbkdf2_hmac(
            'sha256',
            provided_password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        # Compare hashes (constant time comparison untuk security)
        return secrets.compare_digest(pwd_hash, new_hash)
    except Exception as e:
        print(f"Error verifying password: {e}")
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password strength
    
    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    
    Args:
        password: Password to validate
        
    Returns:
        (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password harus minimal 8 karakter"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password harus mengandung minimal 1 huruf besar"
    
    if not re.search(r'[a-z]', password):
        return False, "Password harus mengandung minimal 1 huruf kecil"
    
    if not re.search(r'\d', password):
        return False, "Password harus mengandung minimal 1 angka"
    
    # Check for common weak passwords
    weak_passwords = ['12345678', 'password', 'password123', 'admin123']
    if password.lower() in weak_passwords:
        return False, "Password terlalu lemah, gunakan kombinasi yang lebih kuat"
    
    return True, "Password valid"


def generate_random_password(length: int = 12) -> str:
    """
    Generate random secure password
    
    Args:
        length: Password length (default: 12)
        
    Returns:
        Random password string
    """
    import string
    
    # Ensure password has all required character types
    chars = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(chars) for _ in range(length))
    
    # Make sure it meets requirements
    while True:
        is_valid, _ = validate_password_strength(password)
        if is_valid:
            return password
        password = ''.join(secrets.choice(chars) for _ in range(length))


if __name__ == "__main__":
    # Test password utilities
    print("=== Password Utilities Test ===\n")
    
    # Test 1: Hash and verify
    print("Test 1: Hash and Verify")
    password = "TestPass123"
    hashed = hash_password(password)
    print(f"Original: {password}")
    print(f"Hashed: {hashed}")
    print(f"Verify correct: {verify_password(hashed, password)}")
    print(f"Verify wrong: {verify_password(hashed, 'WrongPass')}")
    print()
    
    # Test 2: Password strength
    print("Test 2: Password Strength Validation")
    test_passwords = [
        "weak",
        "12345678",
        "NoNumbers",
        "nonumbers123",
        "NOLOWERCASE123",
        "ValidPass123"
    ]
    for pwd in test_passwords:
        is_valid, msg = validate_password_strength(pwd)
        print(f"{pwd:20} -> {is_valid:5} | {msg}")
    print()
    
    # Test 3: Generate random password
    print("Test 3: Generate Random Password")
    for i in range(3):
        random_pwd = generate_random_password()
        is_valid, msg = validate_password_strength(random_pwd)
        print(f"Generated: {random_pwd:15} -> Valid: {is_valid}")
