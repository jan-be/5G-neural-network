FROM node:18-alpine as build-stage
WORKDIR /app

COPY package*.json ./
RUN npm ci --force

COPY . .

RUN npm run build


FROM nginx:alpine

COPY --from=build-stage /app/dist/ /files
COPY nginx.conf /etc/nginx/conf.d/default.conf
