# Stage 1: Install Node.js dependencies for Next.js
FROM node:18-alpine AS frontend
WORKDIR /app

# Copy Node.js dependencies and install them
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy frontend code
COPY . .

# Build the Next.js project
RUN npm run build

# Expose the Next.js port
EXPOSE 3000

# Command to run Next.js in development mode
CMD ["npm", "start"]
