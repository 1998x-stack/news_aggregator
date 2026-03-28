#!/usr/bin/env python3
"""
Database initialization script
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db_manager
from loguru import logger


def init_database():
    """Initialize database tables"""
    try:
        logger.info("Starting database initialization...")

        # Create tables
        db_manager.create_tables()

        logger.success("Database initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def reset_database():
    """Reset database (drop and recreate tables)"""
    try:
        logger.warning("Starting database reset...")

        # Drop existing tables
        db_manager.drop_tables()
        logger.info("Existing tables dropped")

        # Create new tables
        db_manager.create_tables()
        logger.success("Database reset completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Database management script")
    parser.add_argument(
        "--reset", action="store_true", help="Reset database (drop and recreate tables)"
    )

    args = parser.parse_args()

    if args.reset:
        success = reset_database()
    else:
        success = init_database()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    main()
