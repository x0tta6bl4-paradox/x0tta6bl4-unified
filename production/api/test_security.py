#!/usr/bin/env python3
"""
Test script for x0tta6bl4 API security features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all security modules can be imported"""
    try:
        from security import security_config, log_security_event
        from auth import authenticate_user, create_user, UserCreate, UserRole
        from validation import InputValidator, validate_request_data
        from encryption import DataEncryption, encrypt_sensitive_data
        print("✓ All security modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_authentication():
    """Test authentication functionality"""
    try:
        from auth import authenticate_user, UserLogin

        # Test login with default user
        user = authenticate_user("user", "user123!")
        if user and user.username == "user":
            print("✓ User authentication works")
            return True
        else:
            print("✗ User authentication failed")
            return False
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False

def test_validation():
    """Test input validation"""
    try:
        from validation import InputValidator

        validator = InputValidator()

        # Test SQL injection detection
        if validator.check_sql_injection("SELECT * FROM users; DROP TABLE users;"):
            print("✓ SQL injection detection works")
        else:
            print("✗ SQL injection detection failed")
            return False

        # Test XSS detection
        if validator.check_xss("<script>alert('xss')</script>"):
            print("✓ XSS detection works")
        else:
            print("✗ XSS detection failed")
            return False

        # Test input sanitization
        sanitized = validator.sanitize_input("test<script>alert('xss')</script>")
        if "<script>" not in sanitized:
            print("✓ Input sanitization works")
            return True
        else:
            print("✗ Input sanitization failed")
            return False

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_encryption():
    """Test encryption functionality"""
    try:
        from encryption import DataEncryption

        enc = DataEncryption()

        # Test encryption/decryption
        test_data = {"secret": "sensitive information", "user_id": 123}
        encrypted = enc.encrypt_data(test_data)
        decrypted = enc.decrypt_data(encrypted)

        if decrypted == test_data:
            print("✓ Encryption/decryption works")
            return True
        else:
            print("✗ Encryption/decryption failed")
            return False

    except Exception as e:
        print(f"✗ Encryption test failed: {e}")
        return False

def test_jwt():
    """Test JWT token functionality"""
    try:
        from auth import create_access_token, verify_token

        test_payload = {"user_id": "test", "username": "testuser", "role": "user"}
        token = create_access_token(test_payload)
        decoded = verify_token(token)

        if decoded and decoded.username == "testuser":
            print("✓ JWT token creation/verification works")
            return True
        else:
            print("✗ JWT token test failed")
            return False

    except Exception as e:
        print(f"✗ JWT test failed: {e}")
        return False

def main():
    """Run all security tests"""
    print("Testing x0tta6bl4 API Security Features")
    print("=" * 50)

    tests = [
        test_imports,
        test_authentication,
        test_validation,
        test_encryption,
        test_jwt
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Security Tests Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All security features are working correctly!")
        return True
    else:
        print("⚠️  Some security features need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)