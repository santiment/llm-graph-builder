# Step 1: Build the React application
FROM node:20 AS build

ARG BACKEND_API_URL="http://llm-graph-builder-backend.stage.san:31080"
ARG REACT_APP_SOURCES="local,youtube,wiki,s3,web"
ARG LLM_MODELS=""
ARG GOOGLE_CLIENT_ID=""
ARG BLOOM_URL="https://workspace-preview.neo4j.io/workspace/explore?connectURL={CONNECT_URL}&search=Show+me+a+graph&featureGenAISuggestions=true&featureGenAISuggestionsInternal=true"
ARG TIME_PER_CHUNK=4
ARG TIME_PER_PAGE=50
ARG LARGE_FILE_SIZE=5242880
ARG CHUNK_SIZE=5242880
ARG CHAT_MODES=""
ARG ENV="DEV"

WORKDIR /app
COPY package.json yarn.lock ./
# RUN yarn add @neo4j-nvl/base @neo4j-nvl/react
RUN yarn install
COPY . ./
RUN BACKEND_API_URL=$BACKEND_API_URL \
    REACT_APP_SOURCES=$REACT_APP_SOURCES \
    LLM_MODELS=$LLM_MODELS \
    GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID \
    BLOOM_URL=$BLOOM_URL \
    TIME_PER_CHUNK=$TIME_PER_CHUNK \
    CHUNK_SIZE=$CHUNK_SIZE \
    ENV=$ENV \
    LARGE_FILE_SIZE=${LARGE_FILE_SIZE} \
    CHAT_MODES=$CHAT_MODES \
    yarn run build

# Step 2: Serve the application using Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
