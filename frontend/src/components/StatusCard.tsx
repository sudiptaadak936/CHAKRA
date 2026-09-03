import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

interface StatusCardProps {
  name: string;
  type: string;
  status: 'connected' | 'disconnected' | 'error' | 'ok' | 'unknown';
  latency?: number;
  details?: string;
}

export const StatusCard: React.FC<StatusCardProps> = ({ name, type, status, latency, details }) => {
  const isHealthy = status === 'connected' || status === 'ok';
  const isDegraded = status === 'error';

  return (
    <div className="bg-gray-900/90 border border-gray-800 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div className="flex items-start justify-between">
        <div>
          <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">{type}</span>
          <h3 className="text-lg font-bold text-white mt-1">{name}</h3>
        </div>
        <div className="flex items-center">
          {isHealthy && <CheckCircle2 className="w-6 h-6 text-emerald-400" />}
          {isDegraded && <AlertTriangle className="w-6 h-6 text-amber-400" />}
          {!isHealthy && !isDegraded && <XCircle className="w-6 h-6 text-rose-500" />}
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-gray-800/80">
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-400">Status</span>
          <span className={`font-semibold capitalize ${isHealthy ? 'text-emerald-400' : 'text-rose-400'}`}>
            {status}
          </span>
        </div>
        {latency !== undefined && (
          <div className="flex justify-between items-center text-sm mt-1">
            <span className="text-gray-400">Latency</span>
            <span className="text-gray-200 font-mono">{latency} ms</span>
          </div>
        )}
        {details && (
          <p className="text-xs text-gray-500 mt-2 truncate" title={details}>
            {details}
          </p>
        )}
      </div>
    </div>
  );
};

