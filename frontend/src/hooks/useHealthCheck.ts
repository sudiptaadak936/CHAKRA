import { useState, useEffect, useCallback } from 'react';
import { HealthResponse, DependenciesHealthResponse } from '../types/health';
import { fetchHealth, fetchDependenciesHealth } from '../services/api';

export function useHealthCheck(intervalMs = 5000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dependencies, setDependencies] = useState<DependenciesHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const checkStatus = useCallback(async () => {
    try {
      const [h, deps] = await Promise.all([
        fetchHealth(),
        fetchDependenciesHealth()
      ]);
      setHealth(h);
      setDependencies(deps);
      setError(null);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to connect to CHAKRA backend.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, intervalMs);
    return () => clearInterval(interval);
  }, [checkStatus, intervalMs]);

  return { health, dependencies, loading, error, lastUpdated, refetch: checkStatus };
}

