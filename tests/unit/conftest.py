"""Unit-test environment bootstrap.

This is loaded by pytest before unit test modules are imported, ensuring the
application settings and database engine are created for a test environment
rather than the developer's local database.
"""
import os

os.environ["DATABASE_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.setdefault("SUPER_ADMIN_EMAILS", "superadmin@test.com")
