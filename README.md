## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- [Python](https://python.org/) (3.8 or higher)
- [Node.js](https://nodejs.org/) (23.x or higher)
- [MongoDB](https://www.mongodb.com/) account
- Git

## 🚀 Getting Started

### 1. MongoDB Setup

Before we run any code we need to set up a MongoDB account and make sure free cluster, since it's not necessary to pay.

1. Create a [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account
2. Create a new cluster
3. In the MongoDB Atlas dashboard:
   - Click "Connect"
   - Choose "Connect your application"
   - Copy your connection string

### 2. Environment Configuration

1. Copy the `config.test.env` file to create a new `.env` file:
   ```bash
   cp config.test.env .env
   ```

2. Fill in the MongoDB settings in your `.env` file:

   Within the MongoDB atlas dashboard, if you press connect, you can find information you need to add into your ``.env``.
   ```
   MONGODB_DB_NAME=your_database_name
   MONGODB_USER=your_username
   MONGODB_PASSWORD=your_password
   MONGODB_HOST=...mongodb.net
   MONGODB_PROTOCOL=mongodb+srv
   # <MONGODB_PROTOCOL>://<MONGODB_USER>:<MONGODB_PASSWORD>@<MONGODB_HOST>/?retryWrites=true&w=majority&appName=<MONGODB_DB_NAME>
   ```

4. Update other environment variables as needed:
   - `SECRET_KEY`: Generate a secure secret key
   - `PROJECT_NAME`: Your project name
   - `SENTRY_DSN`: (Optional) Not needed for now, mainly using Sentry for error tracking

### 3. Backend Setup

1. Create and activate a Python virtual environment \w any virtual environment of choice:
   ```bash
   # Using venv
   python -m venv .venv
   
   # Activate on Windows
   .venv\Scripts\activate
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   ```
   
   ```bash
   # Create a new conda environment
   conda create --name myenv python=3.12
   
   # Activate the conda environment
   conda activate myenv
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

## 🏃‍♂️ Running the Application

Start both frontend and backend with a single command:
```bash
npm run dev # or make frontend
```

For just starting the backend you can look into package manager, and see where just running
```bash
uvicorn api.main:app --reload # or make backend
```

The application will be available at (assuming those ports were open):
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 🔧 API Configuration

The API is configured with the following default settings:
- API prefix: `/api/`
- Token expiration: 30 days
- CORS enabled for:
  - http://localhost:3000
  - http://localhost:8000

## 🔒 Security Notes

1. Always change the `SECRET_KEY` in production
2. Ensure proper CORS settings for production environments
3. Never commit the `.env` file to version control
4. Use secure passwords for MongoDB

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

