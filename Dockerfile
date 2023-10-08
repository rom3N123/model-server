FROM node:16.16-alpine3.16

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

COPY ./dist ./dist

CMD ["npm", "run", "start"]
