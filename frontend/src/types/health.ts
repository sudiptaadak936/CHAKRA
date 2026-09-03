export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  service: string;
}

export interface DependencyStatus {
  status: 'connected' | 'disconnected' | 'error';
  details: string;
  latency_ms: number;
}

export interface DependenciesHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  service: string;
  dependencies: {
    postgres: DependencyStatus;
    neo4j: DependencyStatus;
    redis: DependencyStatus;
  };
}
