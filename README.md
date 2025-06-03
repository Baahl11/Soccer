# Soccer Predictions Platform

🚀 **Plataforma avanzada de predicciones de fútbol con monetización premium**

## 📋 Descripción

Sistema completo de predicciones de fútbol que incluye:
- **Backend FastAPI** con autenticación JWT y sistema de suscripciones
- **Modelos de Machine Learning** para predicciones 1x2, corners, goles
- **Sistema de Value Bets** para identificar apuestas de valor
- **Integración con APIs** de datos deportivos
- **Sistema de monetización** con planes Basic, Pro, Premium y VIP

## 🏗️ Arquitectura

```
Soccer/
├── fastapi_backend/          # API Backend
│   ├── app/
│   │   ├── api/             # Endpoints REST
│   │   ├── core/            # Configuración y seguridad
│   │   ├── models/          # Modelos SQLAlchemy
│   │   └── services/        # Lógica de negocio
├── models/                   # Modelos ML entrenados
├── data_collection/          # Scripts de recolección
└── documentation/           # Documentación técnica

```

## 🚀 Características Principales

### 🔮 Predicciones Avanzadas
- **1x2 Predictions**: Resultado del partido (Local/Empate/Visitante)
- **Corners Predictions**: Predicción de córners totales
- **Goals Predictions**: Predicción de goles con modelos bayesianos
- **Value Bets**: Identificación automática de apuestas de valor

### 💎 Sistema Premium
- **4 Tiers de suscripción**: Basic, Pro, Premium, VIP
- **Límites por tier**: Predicciones diarias, funcionalidades exclusivas
- **Integración Stripe**: Pagos y suscripciones automáticas
- **JWT Authentication**: Seguridad robusta

### 🤖 Machine Learning
- **ELO Rating System**: Sistema avanzado de rating para equipos
- **Modelos Bayesianos**: Para predicciones de goles
- **Feature Engineering**: +50 características por partido
- **Auto-updating**: Modelos que se actualizan automáticamente

## 📊 APIs Integradas

- **API-Football**: Datos en tiempo real de partidos
- **Datos históricos**: Base de datos extensa de temporadas anteriores
- **Live Updates**: Actualizaciones en tiempo real

## 🛠️ Tecnologías

### Backend
- **FastAPI**: Framework web moderno y rápido
- **SQLAlchemy**: ORM para base de datos
- **PostgreSQL/SQLite**: Base de datos
- **JWT**: Autenticación segura
- **Stripe**: Procesamiento de pagos

### Machine Learning
- **scikit-learn**: Modelos de ML
- **pandas**: Manipulación de datos
- **numpy**: Computación numérica
- **joblib**: Persistencia de modelos

### DevOps
- **Uvicorn**: Servidor ASGI
- **Docker**: Containerización (próximamente)
- **GitHub Actions**: CI/CD (próximamente)

## 🚀 Inicio Rápido

### 1. Clonar el repositorio
```bash
git clone https://github.com/Baahl11/Soccer.git
cd Soccer
```

### 2. Configurar entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias
```bash
cd fastapi_backend
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
# Crear .env en fastapi_backend/
DATABASE_URL=sqlite:///./soccer_predictions.db
SECRET_KEY=your-secret-key-here
API_FOOTBALL_KEY=your-api-key-here
```

### 5. Inicializar base de datos
```bash
python create_premium_user.py  # Crear usuario admin de prueba
```

### 6. Ejecutar servidor
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Acceder a la documentación
Visita: http://localhost:8000/docs

## 🔐 Acceso de Prueba

**Usuario Administrador:**
- Email: `admin@soccerpredictions.com`
- Password: `admin123`
- Tier: Premium (acceso completo)

## 📚 Documentación

- [API Documentation](./API_DOCUMENTATION.md)
- [Backend Architecture](./BACKEND_API_ARCHITECTURE.md)
- [Technical Summary](./TECHNICAL_SUMMARY.md)
- [Deployment Guide](./DEPLOYMENT_OPERATIONS_GUIDE.md)

## 🗺️ Roadmap

- [ ] Frontend React/Vue.js
- [ ] Aplicación móvil
- [ ] Notificaciones push
- [ ] Dashboard analytics avanzado
- [ ] Integración con más bookmakers
- [ ] API pública para desarrolladores

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 📞 Contacto

- **GitHub**: [@Baahl11](https://github.com/Baahl11)
- **Email**: contact@soccerpredictions.com

## 🎯 Estado del Proyecto

🟢 **Backend**: Operacional y estable
🟡 **Frontend**: En desarrollo
🟢 **ML Models**: Entrenados y funcionando
🟢 **API Integration**: Activa
🟢 **Payment System**: Configurado

---

⭐ **¡Dale una estrella si te gusta el proyecto!**
