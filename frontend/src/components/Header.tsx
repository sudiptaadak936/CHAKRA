import React from 'react';
import { ShieldAlert, RefreshCw } from 'lucide-react';

interface HeaderProps {
  onRefresh: () => void;
  loading: boolean;
  lastUpdated: Date | null;
}

export const Header: React.FC<HeaderProps> = ({ onRefresh, loading, lastUpdated }) => {
  return (
    <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur px-6 py-4 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-blue-600/20 border border-blue-500/40 rounded-lg text-blue-400">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-wider text-white">CHAKRA</h1>
          <p className="text-xs text-gray-400">Cryptocurrency Fraud Investigation Platform — Step 0 Scaffold</p>
        </div>
      </div>
      <div className="flex items-center space-x-4">
        {lastUpdated && (
          <span className="text-xs text-gray-400">
            Last sync: {lastUpdated.toLocaleTimeString()}
          </span>
        )}
        <button
          onClick={onRefresh}
          disabled={loading}
          className="flex items-center space-x-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-sm font-medium rounded-md border border-gray-700 transition"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-400' : 'text-gray-300'}`} />
          <span>Refresh</span>
        </button>
      </div>
    </header>
  );
};
