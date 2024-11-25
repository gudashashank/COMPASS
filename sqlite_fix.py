import sys
import sqlite3

# Check current sqlite version
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        import subprocess
        # Try installing with --user flag
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pysqlite3-binary"])
        
        # Import and override sqlite3
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        
        # Verify the new version
        import sqlite3
        print(f"Updated SQLite version: {sqlite3.sqlite_version}")
    except Exception as e:
        print(f"Error updating SQLite: {e}")