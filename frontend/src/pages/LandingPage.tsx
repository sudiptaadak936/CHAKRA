import React from 'react';
import { Header } from '../components/Header';
import { StatusCard } from '../components/StatusCard';
import { useHealthCheck } from '../hooks/useHealthCheck';
import { Server } from 'lucide-react';

export const LandingPage: React.FC = () => {
  const { health, dependencies, loading, error, lastUpdated, refetch } = useHealthCheck(5000);

  return (
    <div className="min-h-screen bg-[#0B0F19] text-gray-100 flex flex-col">
      <Header onRefresh={refetch} loading={loading} lastUpdated={lastUpdated} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-10 space-y-8">
        {/* Banner */}
        <div className="bg-gradient-to-r from-blue-950/40 via-gray-900 to-indigo-950/40 border border-blue-900/30 rounded-2xl p-8">
          <div className="max-w-3xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium mb-4">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
              </span>
              <span>System Initialization: Step 0 Foundation Active</span>
            </div>
            <h2 className="text-3xl font-extrabold text-white tracking-tight">
              Cryptocurrency Fraud Investigation Platform
            </h2>
            <p className="mt-2 text-gray-400 text-sm leading-relaxed">
              CHAKRA provides autonomous multi-chain fund tracing, entity clustering, and laundering typology detection for digital asset recovery.
            </p>
          </div>
        </div>

        {/* Global Error Banner */}
        {error && (
          <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-4 text-rose-300 text-sm flex items-center justify-between">
            <span>Backend communication error: {error}</span>
            <button onClick={refetch} className="underline text-xs hover:text-white">Retry Connection</button>
          </div>
        )}

        {/* Core Services Grid */}
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-4 flex items-center space-x-2">
            <Server className="w-4 h-4 text-blue-400" />
            <span>Infrastructure Health & Connectivity</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatusCard
              name="FastAPI Core"
              type="Backend API"
              status={health ? (health.status === 'ok' ? 'ok' : 'error') : 'unknown'}
              details={health?.service || 'chakra-backend'}
            />
            <StatusCard
              name="PostgreSQL 16"
              type="Relational DB"
              status={dependencies?.dependencies.postgres.status || 'unknown'}
              latency={dependencies?.dependencies.postgres.latency_ms}
              details={dependencies?.dependencies.postgres.details}
            />
            <StatusCard
              name="Neo4j Graph DB"
              type="Graph Engine"
              status={dependencies?.dependencies.neo4j.status || 'unknown'}
              latency={dependencies?.dependencies.neo4j.latency_ms}
              details={dependencies?.dependencies.neo4j.details}
            />
            <StatusCard
              name="Redis Store"
              type="Cache & Queue"
              status={dependencies?.dependencies.redis.status || 'unknown'}
              latency={dependencies?.dependencies.redis.latency_ms}
              details={dependencies?.dependencies.redis.details}
            />
          </div>
        </div>

        {/* Architecture Specs */}
        <div className="border border-gray-800 bg-gray-900/40 rounded-xl p-6">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">Planned Pipeline Modules (Incremental Roadmap)</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-400">
            <div className="p-3 bg-gray-950/60 border border-gray-800/80 rounded-lg">
              <span className="text-blue-400 font-semibold block mb-1">Step 1</span>
              Blockchain Ingestion & Multi-Chain RPC Providers
            </div>
            <div className="p-3 bg-gray-950/60 border border-gray-800/80 rounded-lg">
              <span className="text-gray-500 font-semibold block mb-1">Step 2</span>
              Graph Traversal & UTXO / Account Tracing
            </div>
            <div className="p-3 bg-gray-950/60 border border-gray-800/80 rounded-lg">
              <span className="text-gray-500 font-semibold block mb-1">Step 3</span>
              Heuristic Clustering & Exchange Attribution
            </div>
            <div className="p-3 bg-gray-950/60 border border-gray-800/80 rounded-lg">
              <span className="text-gray-500 font-semibold block mb-1">Step 4</span>
              Typology ML & Investigator Copilot
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

