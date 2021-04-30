# d2s-demo-backend

## Prerequisite
Run the following command to start the grobid PDF parsing service:
`docker run -d --init -p 8070:8070 -p 8071:8071 --name grobid lfoppiano/grobid:0.6.1`

## Run this backend
Run locally:
`env DEBUG=True uvicorn app:app --reload --port=5000`

Deploy:
See the doc2slides.service file. Deploy it to systemd.
