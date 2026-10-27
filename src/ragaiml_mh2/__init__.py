def main() -> None:
    import uvicorn

    uvicorn.run("ragaiml_mh2.api:app", host="127.0.0.1", port=8000)
