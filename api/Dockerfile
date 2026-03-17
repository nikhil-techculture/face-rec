FROM node:20-alpine

WORKDIR /app

COPY package.json .
RUN npm install --omit=dev

COPY . .

EXPOSE 1005

CMD ["node", "server.js"]
