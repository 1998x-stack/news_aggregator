"""Offline regression for SQLAlchemy 2.x health probes (no live services)."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class HealthSqlAlchemyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database = types.ModuleType("db.database")
        database.get_db = lambda: None
        cache = types.ModuleType("utils.cache_manager")
        cache.cache_manager = types.SimpleNamespace(redis_client=Mock())
        deps = {"db": types.ModuleType("db"), "db.database": database,
                "utils": types.ModuleType("utils"), "utils.cache_manager": cache}
        path = Path(__file__).resolve().parents[1] / "api/routes/health.py"
        spec = importlib.util.spec_from_file_location("health_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, deps):
            spec.loader.exec_module(module)
        cls.module = module
        cls.cache = cache.cache_manager
        cls.engine = create_engine("sqlite:///:memory:")

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.cache.redis_client.ping.reset_mock()
        self.cache.redis_client.ping.side_effect = None

    def test_details_healthy_with_sqlalchemy_2_session(self):
        with Session(self.engine) as session:
            result = asyncio.run(self.module.health_check_details(db=session))
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["dependencies"]["database"]["status"], "healthy")
        self.assertEqual(result["dependencies"]["redis"]["status"], "healthy")

    def test_readiness_healthy_with_sqlalchemy_2_session(self):
        with Session(self.engine) as session:
            result = asyncio.run(self.module.readiness_check(db=session))
        self.assertEqual(result["status"], "ready")

    def test_details_reports_unavailable_redis(self):
        self.cache.redis_client.ping.side_effect = ConnectionError("offline")
        with Session(self.engine) as session:
            result = asyncio.run(self.module.health_check_details(db=session))
        self.assertEqual(result["status"], "unhealthy")
        self.assertEqual(result["dependencies"]["database"]["status"], "healthy")
        self.assertEqual(result["dependencies"]["redis"]["status"], "unhealthy")

    def test_readiness_reports_unavailable_redis(self):
        self.cache.redis_client.ping.side_effect = ConnectionError("offline")
        with Session(self.engine) as session:
            result = asyncio.run(self.module.readiness_check(db=session))
        self.assertEqual(result["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
