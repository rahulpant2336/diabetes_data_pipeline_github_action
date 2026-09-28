import uvicorn

if __name__ == "__main__":
    # Adjust the module path if your backend file is inside a folder
    # Example: "backend.backend:app" if backend.py is inside a folder named backend
    uvicorn.run(
        "backend.backend:app",   # module:variable
        host="0.0.0.0",          # accessible from outside container
        port=8000,               # default FastAPI port
        reload=True              # auto-reload on code changes
    )
