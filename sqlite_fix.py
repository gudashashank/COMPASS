import sys
import sqlite3
import platform

# Check current sqlite version
print(f"Current SQLite version: {sqlite3.sqlite_version}")

if sqlite3.sqlite_version_info < (3, 35, 0):
    print("SQLite version is too old. Installing pysqlite3-binary...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    
    # Override sqlite3
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    
    # Verify the new version
    import sqlite3
    print(f"Updated SQLite version: {sqlite3.sqlite_version}")