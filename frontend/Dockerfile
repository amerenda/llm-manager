FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
# BUILDTIME changes each push, busting the cache here so index.html is always re-copied from the build stage.
ARG BUILDTIME=0
RUN echo "${BUILDTIME}" > /dev/null
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
