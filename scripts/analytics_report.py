"""Run on demand or with an OS scheduler; output contains aggregates, not questions."""
import json
from intelligence.search_intelligence import SearchIntelligence
from utils.config import settings

def main():
    analytics = SearchIntelligence(settings.search_analytics_db,
        settings.search_analytics_enabled, settings.retention_days)
    print(json.dumps(analytics.summary(), indent=2))

if __name__ == "__main__":
    main()
