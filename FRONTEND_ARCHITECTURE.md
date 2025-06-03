# Frontend Architecture - Soccer Predictions Platform

## 🎯 **COMPONENTE PRINCIPAL: DASHBOARD**

### **Estructura de Componentes React:**

```javascript
src/
├── components/
│   ├── Dashboard/
│   │   ├── DashboardMain.jsx
│   │   ├── MatchCard.jsx
│   │   ├── FilterPanel.jsx
│   │   └── LiveCounter.jsx
│   ├── Predictions/
│   │   ├── PredictionDisplay.jsx
│   │   ├── ConfidenceGauge.jsx
│   │   └── ValueBetIndicator.jsx
│   ├── Layout/
│   │   ├── Header.jsx
│   │   ├── Sidebar.jsx
│   │   └── Footer.jsx
│   └── Common/
│       ├── LoadingSpinner.jsx
│       ├── ErrorBoundary.jsx
│       └── Modal.jsx
├── pages/
│   ├── index.js (Dashboard principal)
│   ├── match/[id].js (Análisis detallado)
│   ├── subscription.js
│   └── profile.js
├── hooks/
│   ├── useMatches.js
│   ├── usePredictions.js
│   └── useFilters.js
├── services/
│   ├── api.js
│   ├── auth.js
│   └── subscription.js
└── utils/
    ├── constants.js
    ├── helpers.js
    └── formatters.js
```

## 📱 **COMPONENTE: MATCH CARD**

```jsx
// components/Dashboard/MatchCard.jsx
import React from 'react';
import { Clock, TrendingUp, Target } from 'lucide-react';

const MatchCard = ({ match }) => {
  const { 
    home_team, 
    away_team, 
    kickoff_time, 
    predictions, 
    confidence,
    league 
  } = match;

  const getConfidenceColor = (conf) => {
    if (conf >= 80) return 'text-green-500 border-green-500';
    if (conf >= 60) return 'text-yellow-500 border-yellow-500';
    return 'text-red-500 border-red-500';
  };

  const getTimeUntilMatch = (kickoff) => {
    const now = new Date();
    const matchTime = new Date(kickoff);
    const diff = matchTime - now;
    
    if (diff < 0) return 'En vivo';
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className={`
      bg-gradient-to-br from-slate-800 to-slate-900 
      rounded-xl p-6 shadow-xl hover:shadow-2xl 
      transition-all duration-300 hover:scale-105
      border-l-4 ${getConfidenceColor(confidence.overall)}
    `}>
      {/* Header del partido */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center space-x-2">
          <span className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-semibold">
            {league.name}
          </span>
          <span className="text-gray-400 text-sm flex items-center">
            <Clock className="w-4 h-4 mr-1" />
            {getTimeUntilMatch(kickoff_time)}
          </span>
        </div>
        <div className="text-right">
          <div className={`text-sm font-bold ${getConfidenceColor(confidence.overall)}`}>
            Confianza: {confidence.overall}%
          </div>
        </div>
      </div>

      {/* Teams */}
      <div className="text-center mb-6">
        <div className="text-xl font-bold text-white mb-2">
          {home_team.name} <span className="text-gray-500">vs</span> {away_team.name}
        </div>
        <div className="text-gray-400 text-sm">
          {new Date(kickoff_time).toLocaleTimeString('es-ES', { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>

      {/* Predicciones principales */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">
            {predictions.match_result.home_win}%
          </div>
          <div className="text-xs text-gray-400">Local</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">
            {predictions.match_result.draw}%
          </div>
          <div className="text-xs text-gray-400">Empate</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">
            {predictions.match_result.away_win}%
          </div>
          <div className="text-xs text-gray-400">Visitante</div>
        </div>
      </div>

      {/* Predicciones adicionales */}
      <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
        <div className="bg-slate-700 rounded-lg p-3">
          <div className="text-gray-300 mb-1">Total Goals</div>
          <div className="text-white font-bold">
            {predictions.goals.total} goles
          </div>
          <div className="text-green-400 text-xs">
            Over 2.5: {predictions.goals.over_2_5}%
          </div>
        </div>
        <div className="bg-slate-700 rounded-lg p-3">
          <div className="text-gray-300 mb-1">Corners</div>
          <div className="text-white font-bold">
            {predictions.corners.total} corners
          </div>
          <div className="text-blue-400 text-xs">
            Over 9.5: {predictions.corners.over_9_5}%
          </div>
        </div>
      </div>

      {/* Value bets indicator */}
      {predictions.value_bets && predictions.value_bets.length > 0 && (
        <div className="bg-gradient-to-r from-yellow-600 to-yellow-500 rounded-lg p-3 mb-4">
          <div className="flex items-center text-black font-semibold text-sm">
            <Target className="w-4 h-4 mr-2" />
            {predictions.value_bets.length} Value Bet{predictions.value_bets.length > 1 ? 's' : ''} Detectado{predictions.value_bets.length > 1 ? 's' : ''}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex space-x-2">
        <button className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors duration-200 text-sm font-semibold">
          Ver Análisis
        </button>
        <button className="bg-slate-600 hover:bg-slate-700 text-white py-2 px-3 rounded-lg transition-colors duration-200">
          <TrendingUp className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default MatchCard;
```

## 🎛️ **COMPONENTE: FILTER PANEL**

```jsx
// components/Dashboard/FilterPanel.jsx
import React, { useState } from 'react';
import { Filter, Star, Clock, TrendingUp } from 'lucide-react';

const FilterPanel = ({ onFiltersChange, activeFilters }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const leagues = [
    { id: 39, name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
    { id: 140, name: 'La Liga', flag: '🇪🇸' },
    { id: 135, name: 'Serie A', flag: '🇮🇹' },
    { id: 78, name: 'Bundesliga', flag: '🇩🇪' },
    { id: 61, name: 'Ligue 1', flag: '🇫🇷' }
  ];

  const confidenceLevels = [
    { value: 'high', label: 'Alta (80%+)', color: 'bg-green-500' },
    { value: 'medium', label: 'Media (60-79%)', color: 'bg-yellow-500' },
    { value: 'low', label: 'Baja (<60%)', color: 'bg-red-500' }
  ];

  const timeRanges = [
    { value: '2h', label: 'Próximas 2h' },
    { value: '6h', label: 'Próximas 6h' },
    { value: '12h', label: 'Próximas 12h' },
    { value: '24h', label: 'Próximas 24h' }
  ];

  const handleFilterChange = (type, value) => {
    const newFilters = { ...activeFilters };
    
    if (type === 'leagues') {
      newFilters.leagues = newFilters.leagues || [];
      if (newFilters.leagues.includes(value)) {
        newFilters.leagues = newFilters.leagues.filter(l => l !== value);
      } else {
        newFilters.leagues.push(value);
      }
    } else {
      newFilters[type] = newFilters[type] === value ? null : value;
    }
    
    onFiltersChange(newFilters);
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-white flex items-center">
          <Filter className="w-5 h-5 mr-2" />
          Filtros Inteligentes
        </h3>
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="md:hidden text-gray-400"
        >
          {isOpen ? '−' : '+'}
        </button>
      </div>

      <div className={`space-y-6 ${isOpen ? 'block' : 'hidden md:block'}`}>
        {/* Ligas */}
        <div>
          <h4 className="text-white font-semibold mb-3 flex items-center">
            🏆 Ligas
          </h4>
          <div className="space-y-2">
            {leagues.map(league => (
              <label key={league.id} className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={activeFilters.leagues?.includes(league.id) || false}
                  onChange={() => handleFilterChange('leagues', league.id)}
                  className="rounded border-gray-600 bg-slate-700 text-blue-600"
                />
                <span className="text-gray-300 text-sm">
                  {league.flag} {league.name}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Nivel de confianza */}
        <div>
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Star className="w-4 h-4 mr-1" />
            Confianza
          </h4>
          <div className="space-y-2">
            {confidenceLevels.map(level => (
              <label key={level.value} className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="confidence"
                  value={level.value}
                  checked={activeFilters.confidence === level.value}
                  onChange={() => handleFilterChange('confidence', level.value)}
                  className="text-blue-600"
                />
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${level.color}`}></div>
                  <span className="text-gray-300 text-sm">{level.label}</span>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Tiempo hasta el partido */}
        <div>
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <Clock className="w-4 h-4 mr-1" />
            Tiempo
          </h4>
          <div className="grid grid-cols-2 gap-2">
            {timeRanges.map(range => (
              <button
                key={range.value}
                onClick={() => handleFilterChange('timeRange', range.value)}
                className={`
                  text-sm py-2 px-3 rounded-lg transition-colors duration-200
                  ${activeFilters.timeRange === range.value 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                  }
                `}
              >
                {range.label}
              </button>
            ))}
          </div>
        </div>

        {/* Filtros especiales */}
        <div>
          <h4 className="text-white font-semibold mb-3 flex items-center">
            <TrendingUp className="w-4 h-4 mr-1" />
            Especiales
          </h4>
          <div className="space-y-2">
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={activeFilters.valueBets || false}
                onChange={() => handleFilterChange('valueBets', !activeFilters.valueBets)}
                className="rounded border-gray-600 bg-slate-700 text-yellow-600"
              />
              <span className="text-gray-300 text-sm">🎯 Solo Value Bets</span>
            </label>
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={activeFilters.highScoring || false}
                onChange={() => handleFilterChange('highScoring', !activeFilters.highScoring)}
                className="rounded border-gray-600 bg-slate-700 text-green-600"
              />
              <span className="text-gray-300 text-sm">⚽ Muchos goles (2.5+)</span>
            </label>
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={activeFilters.bigMatches || false}
                onChange={() => handleFilterChange('bigMatches', !activeFilters.bigMatches)}
                className="rounded border-gray-600 bg-slate-700 text-red-600"
              />
              <span className="text-gray-300 text-sm">🔥 Partidos grandes</span>
            </label>
          </div>
        </div>

        {/* Limpiar filtros */}
        <button
          onClick={() => onFiltersChange({})}
          className="w-full bg-slate-600 hover:bg-slate-700 text-white py-2 px-4 rounded-lg transition-colors duration-200 text-sm"
        >
          Limpiar Filtros
        </button>
      </div>
    </div>
  );
};

export default FilterPanel;
```

## 🏠 **COMPONENTE: DASHBOARD PRINCIPAL**

```jsx
// components/Dashboard/DashboardMain.jsx
import React, { useState, useEffect } from 'react';
import MatchCard from './MatchCard';
import FilterPanel from './FilterPanel';
import LiveCounter from './LiveCounter';
import { useMatches } from '../../hooks/useMatches';
import { Calendar, TrendingUp, Star, Clock } from 'lucide-react';

const DashboardMain = () => {
  const [filters, setFilters] = useState({});
  const { matches, loading, error, refetch } = useMatches(filters);
  const [stats, setStats] = useState({
    totalMatches: 0,
    valueBets: 0,
    highConfidence: 0,
    liveMatches: 0
  });

  useEffect(() => {
    if (matches) {
      setStats({
        totalMatches: matches.length,
        valueBets: matches.filter(m => m.predictions.value_bets?.length > 0).length,
        highConfidence: matches.filter(m => m.confidence.overall >= 80).length,
        liveMatches: matches.filter(m => m.status === 'live').length
      });
    }
  }, [matches]);

  const handleFiltersChange = (newFilters) => {
    setFilters(newFilters);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-white mt-4 text-lg">Cargando predicciones...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 text-lg mb-4">Error al cargar las predicciones</p>
          <button 
            onClick={refetch}
            className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-lg"
          >
            Reintentar
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header con estadísticas */}
      <div className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                ⚽ Soccer Predictions Dashboard
              </h1>
              <p className="text-gray-400">
                Predicciones inteligentes para las próximas 24 horas
              </p>
            </div>
            <LiveCounter />
          </div>

          {/* Estadísticas rápidas */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700 rounded-lg p-4 text-center">
              <Calendar className="w-6 h-6 text-blue-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{stats.totalMatches}</div>
              <div className="text-gray-400 text-sm">Partidos Hoy</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center">
              <TrendingUp className="w-6 h-6 text-yellow-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{stats.valueBets}</div>
              <div className="text-gray-400 text-sm">Value Bets</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center">
              <Star className="w-6 h-6 text-green-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{stats.highConfidence}</div>
              <div className="text-gray-400 text-sm">Alta Confianza</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center">
              <Clock className="w-6 h-6 text-red-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{stats.liveMatches}</div>
              <div className="text-gray-400 text-sm">En Vivo</div>
            </div>
          </div>
        </div>
      </div>

      {/* Contenido principal */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Panel de filtros */}
          <div className="lg:col-span-1">
            <FilterPanel 
              onFiltersChange={handleFiltersChange}
              activeFilters={filters}
            />
          </div>

          {/* Lista de partidos */}
          <div className="lg:col-span-3">
            {matches && matches.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {matches.map((match) => (
                  <MatchCard key={match.id} match={match} />
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-gray-400 text-lg mb-4">
                  No hay partidos que coincidan con los filtros seleccionados
                </p>
                <button 
                  onClick={() => setFilters({})}
                  className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-lg"
                >
                  Limpiar Filtros
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardMain;
```

## 📱 **COMPONENTE: LIVE COUNTER**

```jsx
// components/Dashboard/LiveCounter.jsx
import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';

const LiveCounter = () => {
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [timeUntilNext, setTimeUntilNext] = useState(30 * 60); // 30 minutos en segundos

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeUntilNext(prev => {
        if (prev <= 1) {
          setLastUpdate(new Date());
          return 30 * 60; // Reset a 30 minutos
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-slate-700 rounded-lg p-4 text-center min-w-[200px]">
      <div className="flex items-center justify-center text-green-400 mb-2">
        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
        <span className="text-sm font-semibold">ACTUALIZANDO EN VIVO</span>
      </div>
      <div className="text-white text-lg font-bold">
        {formatTime(timeUntilNext)}
      </div>
      <div className="text-gray-400 text-xs">
        Última actualización: {lastUpdate.toLocaleTimeString('es-ES')}
      </div>
    </div>
  );
};

export default LiveCounter;
```

## 🔗 **HOOK: useMatches**

```javascript
// hooks/useMatches.js
import { useState, useEffect } from 'react';
import { apiService } from '../services/api';

export const useMatches = (filters = {}) => {
  const [matches, setMatches] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchMatches = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const queryParams = new URLSearchParams();
      
      // Agregar filtros a los parámetros
      if (filters.leagues?.length > 0) {
        queryParams.append('leagues', filters.leagues.join(','));
      }
      if (filters.confidence) {
        queryParams.append('confidence', filters.confidence);
      }
      if (filters.timeRange) {
        queryParams.append('timeRange', filters.timeRange);
      }
      if (filters.valueBets) {
        queryParams.append('valueBets', 'true');
      }
      if (filters.highScoring) {
        queryParams.append('highScoring', 'true');
      }
      if (filters.bigMatches) {
        queryParams.append('bigMatches', 'true');
      }

      const response = await apiService.get(`/matches/today?${queryParams.toString()}`);
      setMatches(response.data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching matches:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMatches();
  }, [filters]);

  // Auto-refresh cada 30 minutos
  useEffect(() => {
    const interval = setInterval(fetchMatches, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [filters]);

  return {
    matches,
    loading,
    error,
    refetch: fetchMatches
  };
};
```

## 🎨 **ESTILOS TAILWIND CSS**

```css
/* globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-slate-900 text-white;
  }
}

@layer components {
  .match-card-hover {
    @apply hover:scale-105 hover:shadow-2xl transition-all duration-300;
  }
  
  .confidence-high {
    @apply text-green-500 border-green-500;
  }
  
  .confidence-medium {
    @apply text-yellow-500 border-yellow-500;
  }
  
  .confidence-low {
    @apply text-red-500 border-red-500;
  }
  
  .gradient-card {
    @apply bg-gradient-to-br from-slate-800 to-slate-900;
  }
}
```

---

## 🚀 **SIGUIENTES PASOS**

1. **Esta semana**: Implementar estos componentes base
2. **Próxima semana**: Crear el backend API para alimentar estos componentes
3. **Semana 3**: Integrar sistema de autenticación y suscripciones
4. **Semana 4**: Testing y optimización

¿Te gusta esta arquitectura? ¿Qué componente te gustaría que desarrollemos primero en detalle?
